import supervisely as sly
import functools
import sly_globals as g
import os
import cv2
import ui
import yaml
import torch
from mmdet.apis import inference_detector
import sly_mse_loss
import sly_semantic_head


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value
    return wrapper


@g.my_app.callback("get_output_classes_and_tags")
@sly.timeit
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_custom_inference_settings")
@sly.timeit
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    settings_path = os.path.join(g.root_source_path, "serve/custom_settings.yml")
    sly.logger.info(f"Custom inference settings path: {settings_path}")
    with open(settings_path, 'r') as file:
        default_settings_str = file.read()
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"settings": default_settings_str})


@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "MM Detection Serve",
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.obj_classes),
        "tags_count": len(g.meta.tag_metas),
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


@g.my_app.callback("inference_image_url")
@sly.timeit
@send_error_data
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ext)
    sly.fs.download(image_url, local_image_path)
    results = inference_image_path(image_path=local_image_path, project_meta=g.meta,
                                              context=context, state=state, app_logger=app_logger)
    sly.fs.silent_remove(local_image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=results)


@g.my_app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.my_app.data_dir, sly.rand_str(10) + image_info.name)
    api.image.download_path(image_id, image_path)
    ann_json = inference_image_path(image_path=image_path, project_meta=g.meta,
                                              context=context, state=state, app_logger=app_logger)
    sly.fs.silent_remove(image_path)
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=ann_json)


@g.my_app.callback("inference_batch_ids")
@sly.timeit
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    ids = state["batch_ids"]
    infos = api.image.get_info_by_id_batch(ids)
    paths = []
    for info in infos:
        paths.append(os.path.join(g.my_app.data_dir, sly.rand_str(10) + info.name))
    api.image.download_paths(infos[0].dataset_id, ids, paths)

    result_anns = inference_image_path(image_path=paths, project_meta=g.meta,
                                            context=context, state=state, app_logger=app_logger)
    for image_path in paths:
        sly.fs.silent_remove(image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=result_anns)


def postprocess_one_image_result(result, state, img_size):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
        if isinstance(bbox_result, dict):
            bbox_result, segm_result = bbox_result['ensemble'], segm_result['ensemble']
    else:
        bbox_result, segm_result = result, None
        if isinstance(bbox_result, dict):
            bbox_result = bbox_result['ensemble']
    
    labels = []
    classes = [obj["title"] for obj in g.meta.obj_classes.to_json()]
    if segm_result is None:
        for bboxes, class_name in zip(bbox_result, classes):
            obj_class = g.meta.get_obj_class(class_name)
            for bbox in bboxes:
                top, left, bottom, right, score = int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2]), bbox[4]
                if "confidence_thresh" in state["settings"].keys() and score < state["settings"]["confidence_thresh"]:
                    continue
                rect = sly.Rectangle(top, left, bottom, right)
                conf_tag = sly.Tag(g.meta.get_tag_meta('confidence'), round(float(score), 4))
                rect_label = sly.Label(rect, obj_class, sly.TagCollection([conf_tag]))
                labels.append(rect_label)
    else:
        for bboxes, masks, class_name in zip(bbox_result, segm_result, classes):
            assert len(bbox_result) == len(segm_result)
            obj_class = g.meta.get_obj_class(class_name)
            for bbox, mask in zip(bboxes, masks):
                score = bbox[4]
                if "confidence_thresh" in state["settings"].keys() and score < state["settings"]["confidence_thresh"]:
                    continue
                conf_tag = sly.Tag(g.meta.get_tag_meta('confidence'), round(float(score), 4))
                if mask.any():
                    bitmap = sly.Bitmap(mask)
                    mask_label = sly.Label(bitmap, obj_class, sly.TagCollection([conf_tag]))
                    labels.append(mask_label)

    ann = sly.Annotation(img_size=img_size, labels=labels)
    ann_json = ann.to_json()
    return ann_json


@sly.process_image_roi
def inference_image_path(image_path, project_meta, context, state, app_logger):
    app_logger.debug("Input path(s)", extra={"path(s)": image_path})
    
    if isinstance(image_path, str):
        input_img_shape = cv2.imread(image_path).shape
        result = inference_detector(g.model, image_path)
        torch.cuda.empty_cache()
        result_ann = postprocess_one_image_result(result, state, input_img_shape[:2])
        return result_ann
    else:
        result_anns = []
        for path in image_path:
            input_img_shape = cv2.imread(path).shape
            result = inference_detector(g.model, path)
            torch.cuda.empty_cache()
            ann_json = postprocess_one_image_result(result, state, input_img_shape[:2])
            result_anns.append(ann_json)
        return result_anns



def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.TEAM_ID,
        "context.workspaceId": g.WORKSPACE_ID
    })
    data = {}
    state = {}

    ui.init(data, state)  # init data for UI widgets

    g.my_app.compile_template(g.root_source_path)
    g.my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)