import supervisely as sly
import functools
import sly_globals as g
import os
import cv2
import ui
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
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"settings": {}})


@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "MM Segmentation Serve",
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.obj_classes),
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
    results = inference_image_path(local_image_path, context, state, app_logger)
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
    ann_json = inference_image_path(image_path, context, state, app_logger)
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

    results = []
    # TODO: change 
    for image_path in paths:
        ann_json = inference_image_path(image_path, context, state, app_logger)
        results.append(ann_json)
        sly.fs.silent_remove(image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=results)


# TODO: add crop decorator
def inference_image_path(image_path, context, state, app_logger):
    app_logger.debug("Input path", extra={"path": image_path})

    img = cv2.imread(image_path)
    raw_result = inference_detector(g.model, img)[0]

    labels = []
    classes = [obj["title"] for obj in g.meta.obj_classes.to_json()]

    '''
    for idx, class_name in enumerate(classes):
        class_mask = raw_result == idx
        obj_class = g.meta.get_obj_class(class_name)
        label = sly.Label(sly.Bitmap(class_mask), obj_class)
        labels.append(label)
    '''

    ann = sly.Annotation(img_size=raw_result.shape, labels=labels, )
    ann_json = ann.to_json()

    return ann_json


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