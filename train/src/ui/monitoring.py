import supervisely as sly
from sly_train_progress import init_progress, _update_progress_ui
import sly_globals as g
import os
from functools import partial
from mmcv.cnn.utils import revert_sync_batchnorm
from mmdet.apis import train_detector, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from init_cfg import init_cfg
# import mmcv
# import cv2
# import splits
# import json

# ! required to be left here despite not being used
import sly_imgaugs
import sly_dataset
import sly_logger_hook
import sly_semantic_head
import sly_mse_loss

_open_lnk_name = "open_app.lnk"


def init(data, state):
    
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)
    data["eta"] = None
    state["isValidation"] = False

    init_charts(data, state)
    
    state["collapsedMonitoring"] = True
    state["disabledMonitoring"] = True
    state["doneMonitoring"] = False

    state["started"] = False
    state["preparingData"] = False
    data["outputName"] = None
    data["outputUrl"] = None


def init_chart(title, names, xs, ys, smoothing=None, yrange=None, decimals=None, xdecimals=None):
    series = []
    for name, x, y in zip(names, xs, ys):
        series.append({
            "name": name,
            "data": [[px, py] for px, py in zip(x, y)]
        })
    result = {
        "options": {
            "title": title
        },
        "series": series
    }
    if smoothing is not None:
        result["options"]["smoothingWeight"] = smoothing
    if yrange is not None:
        result["options"]["yaxisInterval"] = yrange
    if decimals is not None:
        result["options"]["decimalsInFloat"] = decimals
    if xdecimals is not None:
        result["options"]["xaxisDecimalsInFloat"] = xdecimals
    return result


def init_charts(data, state):
    state["smoothing"] = 0.6
    # train charts
    state["chartLR"] = init_chart("LR", names=["lr"], xs = [[]], ys = [[]], smoothing=None, decimals=6, xdecimals=2)
    state["chartLoss"] = init_chart("Loss", names=["total", "bbox", "class", "mask", "iou", "rpn_class", "rpn_bbox"], xs=[[]] * 7, ys=[[]] * 7, smoothing=state["smoothing"], decimals=6, xdecimals=2)
    
    # val charts
    state["chartMAP"] = init_chart("Val mAP", names=[], xs=[], ys=[], smoothing=state["smoothing"], decimals=6, xdecimals=2)
    state["chartBoxClassAP"] = init_chart("Val bbox AP", names=[], xs=[], ys=[], smoothing=state["smoothing"], decimals=6, xdecimals=2)
    state["chartMaskClassAP"] = init_chart("Val mask AP", names=[], xs=[], ys=[], smoothing=state["smoothing"], decimals=6, xdecimals=2)
    
    # system charts
    state["chartTime"] = init_chart("Time", names=["time"], xs=[[]], ys=[[]], xdecimals=2)
    state["chartDataTime"] = init_chart("Data Time", names=["data_time"], xs=[[]], ys=[[]], xdecimals=2)
    state["chartMemory"] = init_chart("Memory", names=["memory"], xs=[[]], ys=[[]], xdecimals=2)

@g.my_app.callback("change_smoothing")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def change_smoothing(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "state.chartLoss.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartMAP.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartBoxClassAP.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartMaskClassAP.options.smoothingWeight", "payload": state["smoothing"]},
    ]
    g.api.app.set_fields(g.task_id, fields)


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_artifacts_and_log_progress():
    _save_link_to_ui(g.artifacts_dir, g.my_app.app_url)

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.Progress):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
        else:
            progress.set_current_value(monitor.bytes_read, report=False)
        _update_progress_ui("UploadDir", g.api, g.task_id, progress)

    progress = sly.Progress("Upload directory with training artifacts to Team Files", 0, is_size=True)
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress)

    remote_dir = f"/mmdetection/{g.task_id}_{g.project_info.name}"
    res_dir = g.api.file.upload_directory(g.team_id, g.artifacts_dir, remote_dir, progress_size_cb=progress_cb)
    return res_dir

def init_class_charts_series(state):
    mAP_series = [{"name": "bbox", "data": []}]
    if state["task"] == "instance_segmentation":
        mAP_series.append({
            "name": "mask",
            "data": []
        })

    per_class_series = []
    for class_name in state["selectedClasses"]:
        per_class_series.append({
            "name": class_name,
            "data": []
        })
    fields = [
        {"field": "state.chartMAP.series", "payload": mAP_series},
        {"field": "state.chartBoxClassAP.series", "payload": per_class_series},
        {"field": "state.chartMaskClassAP.series", "payload": per_class_series}
    ]
    g.api.app.set_fields(g.task_id, fields)


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        init_class_charts_series(state)
        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))
        
        cfg = init_cfg(state, state["selectedClasses"], None)
        # dump config
        os.makedirs(os.path.join(g.checkpoints_dir, cfg.work_dir.split('/')[-1]), exist_ok=True)
        cfg.dump(os.path.join(g.checkpoints_dir, cfg.work_dir.split('/')[-1], "config.py"))
        
        # print(f'Ready config:\n{cfg.pretty_text}') # TODO: debug

        # Build the dataset
        datasets = [build_dataset(cfg.data.train)]

        # Build the detector
        model = build_detector(
            cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        # Add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        model = revert_sync_batchnorm(model)
        train_detector(model, datasets, cfg, distributed=False, validate=True)

        # TODO: debug inference
        '''
        with open(splits.val_set_path, "r") as set_file:
            sample = json.load(set_file)["images"][0]["file_name"]
        inference_image_path = os.path.join(g.project_dir, sample)
        img = mmcv.imread(inference_image_path)
        model.cfg = cfg
        result = inference_detector(model, img)
        img = show_result_pyplot(model, img, result)
        cv2.imwrite("/tmp/mmdetection/tmp_seg.png", img)
        '''
        # hide progress bars and eta
        fields = [
            {"field": "data.progressEpoch", "payload": None},
            {"field": "data.progressIter", "payload": None},
            {"field": "data.eta", "payload": None},
        ]
        g.api.app.set_fields(g.task_id, fields)

        remote_dir = upload_artifacts_and_log_progress()
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.doneMonitoring", "payload": True},
            {"field": "state.started", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)

        # stop application
        g.my_app.stop()
        
    except Exception as e:
        g.api.app.set_field(task_id, "state.started", False)
        sly.logger.info(e)
        raise e  # app will handle this error and show modal window

    