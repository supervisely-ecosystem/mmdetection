import errno
import os
import requests
import yaml
import pkg_resources
import sly_globals as g
import supervisely_lib as sly
from mmcv import Config
import init_default_cfg as init_dc

cfg = None

def reload_task(task):
    if task == "detection":
        pretrainedModel = 'TOOD'
    elif task == "instance_segmentation":
        pretrainedModel = 'QueryInst'
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
        {'field': 'data.configLinks', 'payload': configLinks},
        {'field': 'data.modelColumns', 'payload': modelColumns},
        {'field': 'state.selectedModel', 'payload': selectedModel}
    ]
    g.api.app.set_fields(g.task_id, fields)


def init(data, state):
    state['pretrainedModel'] = 'TOOD'
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
    state["useAuxiliaryHead"] = True
    state["weightsInitialization"] = "pretrained"  # "custom"
    state["collapsedModels"] = True
    state["disabledModels"] = True
    state["weightsPath"] = ""
    data["doneModels"] = False
    state["loadingModel"] = False

    # default hyperparams that may be reassigned from model default params
    init_dc.init_default_cfg_args(state)

    sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress6", "Download weights", is_size=True,
                                min_report_percent=5).init_data(data)


def get_pretrained_models(task="detection", return_metrics=False):
    model_yamls = sly.json.load_json_file(os.path.join(g.root_source_dir, "models", f"{task}_meta.json"))
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
    file_info = g.api.file.get_info_by_path(g.team_id, remote_path)
    if file_info is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), remote_path)
    progress.set_total(file_info.sizeb)
    g.api.file.download(g.team_id, remote_path, local_path, g.my_app.cache,
                        progress.increment)
    progress.reset_and_update()

    sly.logger.info(f"{remote_path} has been successfully downloaded",
                    extra={"weights": local_path})


def download_custom_config(state):
    progress = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress6", "Download config", is_size=True,
                                           min_report_percent=5)

    weights_remote_dir = os.path.dirname(state["weightsPath"])
    model_config_local_path = os.path.join(g.checkpoints_dir, g.my_app.data_dir.split('/')[-1], 'custom_loaded_config.py')

    config_remote_dir = os.path.join(weights_remote_dir, f'config.py')
    if g.api.file.exists(g.team_id, config_remote_dir):
        download_sly_file(config_remote_dir, model_config_local_path, progress)
    return model_config_local_path


@g.my_app.callback("download_weights")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_weights(api: sly.Api, task_id, context, state, app_logger):
    progress = sly.app.widgets.ProgressBar(g.task_id, g.api, "data.progress6", "Download weights", is_size=True,
                                           min_report_percent=5)
    with_semantic_masks = False
    model_config_local_path = None
    try:
        if state["weightsInitialization"] == "custom":
            # raise NotImplementedError
            weights_path_remote = state["weightsPath"]
            if not weights_path_remote.endswith(".pth"):
                raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                                 f"Supported: '.pth'")

            g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
            if sly.fs.file_exists(g.local_weights_path):
                os.remove(g.local_weights_path)

            download_sly_file(weights_path_remote, g.local_weights_path, progress)
            model_config_local_path = download_custom_config(state)

        else:
            checkpoints_by_model = get_pretrained_models(state["task"])[state["pretrainedModel"]]["checkpoints"]
            selected_model = next(item for item in checkpoints_by_model
                                  if item["name"] == state["selectedModel"][state["pretrainedModel"]])

            weights_url = selected_model.get('weights')
            config_file = selected_model.get('config_file')
            with_semantic_masks = selected_model.get('semantic')
            if weights_url is not None:
                g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
                model_config_local_path = os.path.join(g.root_source_dir, config_file)
                # TODO: check that pretrained weights are exist on remote server
                if sly.fs.file_exists(g.local_weights_path) is False:
                    response = requests.head(weights_url, allow_redirects=True)
                    sizeb = int(response.headers.get('content-length', 0))
                    progress.set_total(sizeb)
                    os.makedirs(os.path.dirname(g.local_weights_path), exist_ok=True)
                    sly.fs.download(weights_url, g.local_weights_path, g.my_app.cache, progress.increment)
                    progress.reset_and_update()
                sly.logger.info("Pretrained weights has been successfully downloaded",
                                extra={"weights": g.local_weights_path})



    except Exception as e:
        progress.reset_and_update()
        raise e

    fields = [
        {"field": "state.loadingModel", "payload": False},
        {"field": "data.doneModels", "payload": True},
        {"field": "state.collapsedClasses", "payload": False},
        {"field": "state.disabledClasses", "payload": False},
        {"field": "state.activeStep", "payload": 4},
    ]

    global cfg
    if model_config_local_path is None:
        raise ValueError("Model config file not found!")
    cfg = Config.fromfile(model_config_local_path)
    if state["weightsInitialization"] != "custom":
        cfg.pretrained_model = state["pretrainedModel"]
        cfg.with_semantic_masks = with_semantic_masks

    print(f'Initial config:\n{cfg.pretty_text}') # TODO: debug
    params = init_dc.rewrite_default_cfg_args(cfg, state)
    fields.extend(params)

    g.api.app.set_fields(g.task_id, fields)

def restart(data, state):
    data["doneModels"] = False