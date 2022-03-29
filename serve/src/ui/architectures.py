import errno
import os
import sys
import yaml
import requests
import pkg_resources
import sly_globals as g
import supervisely_lib as sly
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.datasets import *
from supervisely.app.v1.widgets.progress_bar import ProgressBar


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def reload_task(task):
    if task == "detection":
        pretrainedModel = 'TOOD'
        taskTitle = 'Object detection'
    elif task == "instance_segmentation":
        pretrainedModel = 'QueryInst'
        taskTitle = 'Instance segmentation'
    pretrainedModels, metrics = get_pretrained_models(task, return_metrics=True)
    model_select_info = []
    for model_name, params in pretrainedModels.items():
        model_select_info.append({
            "name": model_name,
            "paper_from": params["paper_from"],
            "year": params["year"]
        })
    pretrainedModelsInfo = model_select_info
    configLinks = {model_name: params["config_url"] for model_name, params in pretrainedModels.items()}

    modelColumns = get_table_columns(metrics)

    selectedModel = {pretrained_model: pretrainedModels[pretrained_model]["checkpoints"][0]['name']
                              for pretrained_model in pretrainedModels.keys()}
 
    fields = [
        {'field': 'state.pretrainedModel', 'payload': pretrainedModel},
        {'field': 'data.pretrainedModels', 'payload': pretrainedModels},
        {'field': 'data.pretrainedModelsInfo', 'payload': pretrainedModelsInfo},
        {'field': 'data.taskTitle', 'payload': taskTitle},
        {'field': 'data.configLinks', 'payload': configLinks},
        {'field': 'data.modelColumns', 'payload': modelColumns},
        {'field': 'state.selectedModel', 'payload': selectedModel}
    ]
    g.api.app.set_fields(g.TASK_ID, fields)


def init(data, state):
    state['pretrainedModel'] = 'TOOD'
    sly.logger.info("Reading model data from configs...")
    data["pretrainedModels"], metrics = get_pretrained_models("detection", return_metrics=True)
    model_select_info = []
    for model_name, params in data["pretrainedModels"].items():
        model_select_info.append({
            "name": model_name,
            "paper_from": params["paper_from"],
            "year": params["year"]
        })
    data["pretrainedModelsInfo"] = model_select_info
    data["configLinks"] = {model_name: params["config_url"] for model_name, params in data["pretrainedModels"].items()}

    data["modelColumns"] = get_table_columns(metrics)

    state["selectedModel"] = {pretrained_model: data["pretrainedModels"][pretrained_model]["checkpoints"][0]['name']
                              for pretrained_model in data["pretrainedModels"].keys()}

    sly.logger.info("Model data is ready.")
    state["weightsInitialization"] = "pretrained"  # "custom"
    state["collapsedModels"] = True
    state["disabledModels"] = True
    state["weightsPath"] = ""
    state["loadingModel"] = False
    state["device"] = "cuda:0"
    data["taskTitle"] = "Object Detection"

    ProgressBar(g.TASK_ID, g.api, "data.progressWeights", "Download weights", is_size=True,
                                min_report_percent=5).init_data(data)


def get_pretrained_models(task="detection", return_metrics=False):
    model_yamls = sly.json.load_json_file(os.path.join(g.models_configs_dir, f"{task}_meta.json"))
    model_config = {}
    all_metrics = []
    for model_meta in model_yamls:
        with open(os.path.join(g.configs_dir, model_meta["yml_file"]), "r") as stream:
            model_info = yaml.safe_load(stream)
            model_config[model_meta["model_name"]] = {}
            model_config[model_meta["model_name"]]["checkpoints"] = []
            model_config[model_meta["model_name"]]["paper_from"] = model_meta["paper_from"]
            model_config[model_meta["model_name"]]["year"] = model_meta["year"]
            mmdet_ver = pkg_resources.get_distribution("mmdet").version
            model_config[model_meta["model_name"]]["config_url"] = f"https://github.com/open-mmlab/mmdetection/tree/v{mmdet_ver}/configs/" + model_meta["yml_file"].split("/")[0]
            checkpoint_keys = []
            for model in model_info["Models"]:
                checkpoint_info = {}
                if "exclude" in model_meta.keys():
                    if model_meta["exclude"].endswith("*"):
                        if model["Name"].startswith(model_meta["exclude"][:-1]):
                            continue
                checkpoint_info["semantic"] = False
                if "semantic" in model_meta.keys():
                    if model_meta["semantic"] == "*":
                        checkpoint_info["semantic"] = True
                    elif model_meta["semantic"].startswith("*") and model_meta["semantic"].endswith("*"):
                        if model_meta["semantic"][1:-1] in model["Name"]:
                            checkpoint_info["semantic"] = True
                    elif model_meta["semantic"].startswith("*") and model["Name"].endswith(model_meta["semantic"][1:]):
                        checkpoint_info["semantic"] = True
                    elif model_meta["semantic"].endswith("*") and model_meta["semantic"].startswith("!"):
                        if not model["Name"].startswith(model_meta["semantic"][1:-1]):
                            checkpoint_info["semantic"] = True
                    
                checkpoint_info["name"] = model["Name"]
                try:
                    checkpoint_info["inference_time"] = model["Metadata"]["inference time (ms/im)"][0]["value"]
                except KeyError:
                    checkpoint_info["inference_time"] = "-"
                try:
                    checkpoint_info["resolution"] = model["Metadata"]["inference time (ms/im)"][0]["resolution"]
                except KeyError:
                    checkpoint_info["resolution"] = "-"
                try:
                    checkpoint_info["epochs"] = model["Metadata"]["Epochs"]
                except KeyError:
                    checkpoint_info["epochs"] = "-"
                try:
                    checkpoint_info["training_memory"] = model["Metadata"]["Training Memory (GB)"]
                except KeyError:
                    checkpoint_info["training_memory"] = "-"
                checkpoint_info["config_file"] = model["Config"]
                for result in model["Results"]:
                    if (task == "detection" and result["Task"] == "Object Detection") or \
                       (task == "instance_segmentation" and result["Task"] == "Instance Segmentation"):
                        checkpoint_info["dataset"] = result["Dataset"]
                        for metric_name, metric_val in result["Metrics"].items():
                            if metric_name not in all_metrics:
                                all_metrics.append(metric_name)
                            checkpoint_info[metric_name] = metric_val
                try:
                    checkpoint_info["weights"] = model["Weights"]
                except KeyError as e:
                    sly.logger.info(f'Weights not found. Model: {model_meta["model_name"]}, checkpoint: {checkpoint_info["name"]}')
                    continue
                for key in checkpoint_info.keys():
                    checkpoint_keys.append(key)
                model_config[model_meta["model_name"]]["checkpoints"].append(checkpoint_info)
            model_config[model_meta["model_name"]]["all_keys"] = checkpoint_keys
    if return_metrics:
        return model_config, all_metrics
    return model_config


def get_table_columns(metrics):
    columns = [
        {"key": "name", "title": " ", "subtitle": None},
        {"key": "dataset", "title": "Dataset", "subtitle": None},
        {"key": "inference_time", "title": "Inference time", "subtitle": "(ms/im)"},
        {"key": "resolution", "title": "Input size", "subtitle": "(H, W)"},
        {"key": "epochs", "title": "Epochs", "subtitle": None},
        {"key": "training_memory", "title": "Training memory", "subtitle": "GB"},
    ]
    for metric in metrics:
        columns.append({"key": metric, "title": f"{metric} score", "subtitle": None})
    return columns


def download_sly_file(remote_path, local_path, progress):
    file_info = g.api.file.get_info_by_path(g.TEAM_ID, remote_path)
    if file_info is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), remote_path)
    progress.set_total(file_info.sizeb)
    g.api.file.download(g.TEAM_ID, remote_path, local_path, g.my_app.cache,
                        progress.increment)
    progress.reset_and_update()

    sly.logger.info(f"{remote_path} has been successfully downloaded",
                    extra={"weights": local_path})


def download_custom_config(state):
    weights_remote_dir = os.path.dirname(state["weightsPath"])
    model_config_local_path = os.path.join(g.my_app.data_dir, 'config.py')

    config_remote_dir = os.path.join(weights_remote_dir, f'config.py')
    if g.api.file.exists(g.TEAM_ID, config_remote_dir):
        download_sly_file(config_remote_dir, model_config_local_path)
    return model_config_local_path


def download_weights(state):
    progress = ProgressBar(g.TASK_ID, g.api, "data.progressWeights", "Download weights", is_size=True,
                                           min_report_percent=5)
    if state["weightsInitialization"] == "custom":
        weights_path_remote = state["weightsPath"]
        if not weights_path_remote.endswith(".pth"):
            raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                                f"Supported: '.pth'")

        g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
        if sly.fs.file_exists(g.local_weights_path):
            os.remove(g.local_weights_path)

        download_sly_file(weights_path_remote, g.local_weights_path, progress)
        g.model_config_local_path = download_custom_config(state)

    else:
        checkpoints_by_model = get_pretrained_models(state["task"])[state["pretrainedModel"]]["checkpoints"]
        selected_model = next(item for item in checkpoints_by_model
                                if item["name"] == state["selectedModel"][state["pretrainedModel"]])

        weights_url = selected_model.get('weights')
        config_file = selected_model.get('config_file')
        if weights_url is not None:
            g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
            g.model_config_local_path = os.path.join(g.root_source_path, config_file)
            if sly.fs.file_exists(g.local_weights_path) is False:
                response = requests.head(weights_url, allow_redirects=True)
                sizeb = int(response.headers.get('content-length', 0))
                progress.set_total(sizeb)
                os.makedirs(os.path.dirname(g.local_weights_path), exist_ok=True)
                sly.fs.download(weights_url, g.local_weights_path, g.my_app.cache, progress.increment)
                progress.reset_and_update()
            sly.logger.info("Pretrained weights has been successfully downloaded",
                            extra={"weights": g.local_weights_path})


def init_model_and_cfg(state):
    g.cfg = Config.fromfile(g.model_config_local_path)
    if 'pretrained' in g.cfg.model:
        g.cfg.model.pretrained = None
    elif 'init_cfg' in g.cfg.model.backbone:
        g.cfg.model.backbone.init_cfg = None
    g.cfg.model.train_cfg = None
    model = build_detector(g.cfg.model, test_cfg=g.cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, g.local_weights_path, map_location='cpu')
    if state["weightsInitialization"] == "custom":
        classes = g.cfg.checkpoint_config.meta.CLASSES
        if "segm" in g.cfg.evaluation.metric:
            obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in classes]
        else:
            obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in classes]
    else:
        dataset_class_name = g.cfg.dataset_type
        classes = str_to_class(dataset_class_name).CLASSES
        if state["task"] == "detection":
            obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in classes]
        elif state["task"] == "instance_segmentation":
            obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in classes]

    model.CLASSES = classes
    model.cfg = g.cfg
    model.to(g.device)
    model.eval()
    model = revert_sync_batchnorm(model)
    g.model = model

    tags = [sly.TagMeta('confidence', sly.TagValueType.ANY_NUMBER)]

    g.meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes),
                             tag_metas=sly.TagMetaCollection(tags))


@g.my_app.callback("run")
@g.my_app.ignore_errors_and_show_dialog_window()
def init_model(api: sly.Api, task_id, context, state, app_logger):
    g.remote_weights_path = state["weightsPath"]
    g.device = state["device"]
    download_weights(state)
    init_model_and_cfg(state)
    fields = [
        {"field": "state.loadingModel", "payload": False},
        {"field": "state.deployed", "payload": True},
    ]
    g.api.app.set_fields(g.TASK_ID, fields)
    sly.logger.info("Model has been successfully deployed")
