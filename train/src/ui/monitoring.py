import supervisely as sly
from sly_train_progress import init_progress, _update_progress_ui
import sly_globals as g
import os
import shutil
import cv2
import numpy as np
from functools import partial
import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmdet.apis import train_detector, set_random_seed, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
# from init_cfg import init_cfg
from mmcv import Config
import splits
import json
import matplotlib.pyplot as plt
import architectures

# ! required to be left here despite not being used
import sly_imgaugs
import sly_dataset
import sly_logger_hook

_open_lnk_name = "open_app.lnk"


def init(data, state):
    '''
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)
    data["eta"] = None
    state["isValidation"] = False

    init_charts(data, state)
    '''
    state["collapsedMonitoring"] = True
    state["disabledMonitoring"] = True
    state["doneMonitoring"] = False

    state["started"] = False
    state["preparingData"] = False
    data["outputName"] = None
    data["outputUrl"] = None


def init_chart(title, names, xs, ys, smoothing=None, yrange=None, decimals=None, xdecimals=None, metric=None):
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
    if len(names) > 0:
        result["series"] = series
    if metric is not None:
        result["metric"] = metric
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
    state["chartLR"] = init_chart("LR", names=["lr"], xs = [[]], ys = [[]], smoothing=None,
                                 # yrange=[state["lr"] - state["lr"] / 2.0, state["lr"] + state["lr"] / 2.0],
                                 decimals=6, xdecimals=2)
    state["chartTrainLoss"] = init_chart("Train Loss", names=["loss"], xs=[[]], ys=[[]], smoothing=state["smoothing"], decimals=6, xdecimals=2)
    state["mean_charts"] = {}
    for metric in data["availableMetrics"]:
        state["mean_charts"][f"chartVal_{metric}"] = init_chart(f"Val {metric}", metric=metric, names=[metric], xs=[[]], ys=[[]], smoothing=state["smoothing"])
    state["class_charts"] = {}
    for metric in data["availableMetrics"]:
        state["class_charts"][f"chartVal_{metric[1:]}"] = init_chart(f"Val {metric[1:]}", names=[], metric=metric, xs=[], ys=[], smoothing=state["smoothing"])

    state["chartTime"] = init_chart("Time", names=["time"], xs=[[]], ys=[[]], xdecimals=2)
    state["chartDataTime"] = init_chart("Data Time", names=["data_time"], xs=[[]], ys=[[]], xdecimals=2)
    state["chartMemory"] = init_chart("Memory", names=["memory"], xs=[[]], ys=[[]], xdecimals=2)

@g.my_app.callback("change_smoothing")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def change_smoothing(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "state.chartTrainLoss.options.smoothingWeight", "payload": state["smoothing"]}
    ]
    for metric in state["evalMetrics"]:
        fields.extend([
            {"field": f"state.mean_charts.chartVal_{metric}.options.smoothingWeight", "payload": state["smoothing"]},
            {"field": f"state.class_charts.chartVal_{metric[1:]}.options.smoothingWeight", "payload": state["smoothing"]}
        ])
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
    classes = state["selectedClasses"] + ["__bg__"]
    series = []
    for class_name in classes:
        series.append({
            "name": class_name,
            "data": []
        })
    fields = [
        {"field": "state.preparingData", "payload": True}
    ]
    for metric_name in state["evalMetrics"]:
        fields.extend([
            {"field": f"state.class_charts.chartVal_{metric_name[1:]}.series", "payload": series}
        ])
    g.api.app.set_fields(g.task_id, fields)


def prepare_segmentation_data(state, img_dir, ann_dir):
    temp_project_seg_dir = g.project_seg_dir + "_temp"
    sly.Project.to_segmentation_task(g.project_dir, temp_project_seg_dir, target_classes=state["selectedClasses"])
    shutil.rmtree(g.project_dir)
    project_seg_temp = sly.Project(temp_project_seg_dir, sly.OpenMode.READ)
    classes_json = project_seg_temp.meta.obj_classes.to_json()
    classes = [obj["title"] for obj in classes_json if obj["title"]]
    '''
    ignore_index = None
    for ind, class_name in enumerate(classes):
        if class_name == "__bg__":
            ignore_index = ind
            break
    '''
    palette = [obj["color"].lstrip('#') for obj in classes_json]
    # hex to rgb
    palette = [[int(color[i:i + 2], 16) for i in (0, 2, 4)] for color in palette]

    datasets = os.listdir(temp_project_seg_dir)
    os.makedirs(os.path.join(g.project_seg_dir, img_dir), exist_ok=True)
    os.makedirs(os.path.join(g.project_seg_dir, ann_dir), exist_ok=True)
    for dataset in datasets:
        if not os.path.isdir(os.path.join(temp_project_seg_dir, dataset)):
            if dataset == "meta.json":
                shutil.move(os.path.join(temp_project_seg_dir, "meta.json"), g.project_seg_dir)
            continue
        # convert masks to required format and save to general ann_dir
        mask_files = os.listdir(os.path.join(temp_project_seg_dir, dataset, ann_dir))
        for mask_file in mask_files:
            mask = cv2.imread(os.path.join(temp_project_seg_dir, dataset, ann_dir, mask_file))[:, :, ::-1]
            result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
            # human masks to machine masks
            for color_idx, color in enumerate(palette):
                colormap = np.where(np.all(mask == color, axis=-1))
                result[colormap] = color_idx
            cv2.imwrite(os.path.join(g.project_seg_dir, ann_dir, mask_file), result)

        imgfiles_to_move = os.listdir(os.path.join(temp_project_seg_dir, dataset, img_dir))
        for filename in imgfiles_to_move:
            shutil.move(os.path.join(temp_project_seg_dir, dataset, img_dir, filename),
                        os.path.join(g.project_seg_dir, img_dir))

    shutil.rmtree(temp_project_seg_dir)
    g.api.app.set_field(g.task_id, "state.preparingData", False)
    return classes, palette


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    # init_class_charts_series(state)
    try:
        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))
        '''
        img_dir = "img"
        ann_dir = "seg"
        classes, palette = prepare_segmentation_data(state, img_dir, ann_dir)

        cfg = init_cfg(state, img_dir, ann_dir, classes, palette)

        os.makedirs(os.path.join(g.checkpoints_dir, cfg.work_dir.split('/')[-1]), exist_ok=True)
        cfg.dump(os.path.join(g.checkpoints_dir, cfg.work_dir.split('/')[-1], "config.py"))
        '''
        g.project_det_dir = g.project_dir
        project_det = sly.Project(g.project_det_dir, sly.OpenMode.READ)
        g.project_det_meta = g.project_meta
        classes_json = project_det.meta.obj_classes.to_json()
        classes = [obj["title"] for obj in classes_json if obj["title"]]
        
        # cfg_path = os.path.join(g.root_source_dir, 'configs', 'faster_rcnn', 'faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
        # cfg = Config.fromfile(cfg_path)
        cfg = architectures.cfg

        # Modify dataset type and path
        cfg.dataset_type = 'SuperviselyDataset'
        cfg.data_root = g.project_det_dir
        cfg.data.samples_per_gpu = 2

        cfg.data.train.type = cfg.dataset_type
        cfg.data.train.data_root = cfg.data_root
        cfg.data.train.ann_file = splits.train_set_path
        cfg.data.train.img_prefix = None
        cfg.data.train.seg_prefix = None
        cfg.data.train.proposal_file = None
        cfg.data.train.test_mode = False
        cfg.data.train.classes = classes
        cfg.data.train.task = state["task"]

        cfg.data.val.type = cfg.dataset_type
        cfg.data.val.data_root = cfg.data_root
        cfg.data.val.ann_file = splits.val_set_path
        cfg.data.val.img_prefix = None
        cfg.data.val.seg_prefix = None
        cfg.data.val.proposal_file = None
        cfg.data.val.test_mode = False
        cfg.data.val.classes = classes
        cfg.data.val.task = state["task"]
        cfg.data.val.samples_per_gpu = 2

        cfg.data.test.type = cfg.dataset_type
        cfg.data.test.data_root = cfg.data_root
        cfg.data.test.ann_file = None
        cfg.data.test.img_prefix = None
        cfg.data.test.seg_prefix = None
        cfg.data.test.proposal_file = None
        cfg.data.test.test_mode = True
        cfg.data.test.classes = classes
        cfg.data.test.task = state["task"]

        # modify num classes of the model in box head
        if hasattr(cfg.model, "roi_head"):
            cfg.model.roi_head.bbox_head.num_classes = 2
        
        elif hasattr(cfg.model, "bbox_head") and not isinstance(cfg.model.bbox_head, list):
            cfg.model.bbox_head.num_classes = 2
        else:
            raise ValueError("No bbox head")
        # We can still use the pre-trained Mask RCNN model though we do not need to
        # use the mask branch
        # cfg.load_from = os.path.join(g.root_source_dir, 'checkpoints', 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')
        cfg.load_from = g.local_weights_path

        # Set up working dir to save files and logs.
        cfg.work_dir = g.my_app.data_dir

        # The original learning rate (LR) is set for 8-GPU training.
        # We divide it by 8 since we only use one GPU.
        cfg.optimizer.lr = 0.02 / 8
        cfg.lr_config.warmup = None
        cfg.log_config.interval = 1
        cfg.log_config.hooks = [
        dict(type='SuperviselyLoggerHook', by_epoch=False)
    ]

        # Change the evaluation metric since we use customized dataset.
        cfg.evaluation.metric = 'mAP'
        # We can set the evaluation interval to reduce the evaluation times
        cfg.evaluation.interval = 12
        # We can set the checkpoint saving interval to reduce the storage cost
        cfg.checkpoint_config.interval = 12

        # Set seed thus the results are more reproducible
        cfg.seed = 0
        set_random_seed(0, deterministic=False)
        cfg.gpu_ids = range(1)

        # We can initialize the logger for training and have a look
        # at the final config used for training
        print(f'Config:\n{cfg.pretty_text}')

        # Build the dataset
        datasets = [build_dataset(cfg.data.train)]

        # Build the detector
        model = build_detector(
            cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        # Add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        model = revert_sync_batchnorm(model)
        # Create work_dir
        os.makedirs(os.path.abspath(cfg.work_dir), exist_ok=True)
        train_detector(model, datasets, cfg, distributed=False, validate=True)

        with open(splits.val_set_path, 'r') as set_file:
            sample = json.load(set_file)[0]
        inference_image_path = os.path.join(g.project_det_dir, sample["dataset_name"], "img", sample["item_name"])
        img = mmcv.imread(inference_image_path)

        model.cfg = cfg
        result = inference_detector(model, img)

        img = show_result_pyplot(model, img, result)
        cv2.imwrite("/tmp/mmdetection/tmp.png", img)

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
        '''
    except Exception as e:
        g.api.app.set_field(task_id, "state.started", False)
        sly.logger.info(e)
        raise e  # app will handle this error and show modal window

    # stop application
    g.my_app.stop()