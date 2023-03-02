import errno
import os
import sys
import yaml
import requests
import pkg_resources
import sly_globals as g
import torch
import supervisely_lib as sly
from supervisely.app.v1.widgets.progress_bar import ProgressBar


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
    device_values = ["cpu"]
    device_names = ["CPU"]
    if torch.cuda.is_available():
        gpus = torch.cuda.device_count()
        for i in range(gpus):
            device_values.append(f"cuda:{i}")
            device_names.append(f"{torch.cuda.get_device_name(i)} (cuda:{i})")

    data["available_device_names"] = device_names
    data["available_device_values"] = device_values
    state["device"] = device_values[0]
    data["taskTitle"] = "Object Detection"

    ProgressBar(g.TASK_ID, g.api, "data.progressWeights", "Download weights", is_size=True,
                                min_report_percent=5).init_data(data)

