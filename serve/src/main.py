import os
import json
import shutil
import sys
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import List, Any, Dict, Union
from pathlib import Path
import numpy as np
import yaml
from dotenv import load_dotenv
import torch
import supervisely as sly
import supervisely.app.widgets as Widgets
import supervisely.nn.inference.gui as GUI
from gui import MMDetectionGUI
from supervisely.nn.prediction_dto import PredictionBBox, PredictionMask
import pkg_resources
from collections import OrderedDict
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.datasets import *
from mmdet.apis import inference_detector
import sly_mse_loss
import sly_semantic_head

root_source_path = str(Path(__file__).parents[2])
app_source_path = str(Path(__file__).parents[1])
load_dotenv(os.path.join(app_source_path, "local.env"))
load_dotenv(os.path.expanduser("~/supervisely.env"))

use_gui_for_local_debug = bool(int(os.environ.get("USE_GUI", "1")))

models_meta_path = os.path.join(root_source_path, "models", "detection_meta.json")

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)



def get_message_from_exception(exception):
    try:
        json_text = exception.args[0].response.text
        info = json.loads(json_text)
        exc_message = info.get("message", repr(exception))
    except:
        exc_message = repr(exception)
    return exc_message


configs_dir = os.path.join(root_source_path, "configs")
mmdet_ver = pkg_resources.get_distribution("mmdet").version
if os.path.isdir(f"/tmp/mmdet/mmdetection-{mmdet_ver}"):
    if os.path.isdir(configs_dir):
        shutil.rmtree(configs_dir)
    sly.logger.info(f"Getting model configs of current mmdetection version {mmdet_ver}...")
    shutil.copytree(f"/tmp/mmdet/mmdetection-{mmdet_ver}/configs", configs_dir)
    models_cnt = len(os.listdir(configs_dir)) - 1
    sly.logger.info(f"Found {models_cnt} models in {configs_dir} directory.")

class MMDetectionModel(sly.nn.inference.InstanceSegmentation):
    def load_on_device(
        self,
        model_dir: str, 
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ) -> None:
        self.device = device
        if self.gui is not None:
            self.task_type = self.gui.get_task_type()
            model_source = self.gui.get_model_source()
            if model_source == "Pretrained models":
                selected_model = self.gui.get_checkpoint_info()
                weights_path, config_path = self.download_pretrained_files(selected_model, model_dir)
            elif model_source == "Custom models":
                custom_weights_link = self.gui.get_custom_link()
                weights_path, config_path = self.download_custom_files(custom_weights_link, model_dir)
        else:
            # for local debug without GUI only
            self.task_type = task_type
            model_source = "Pretrained models"
            weights_path, config_path = self.download_pretrained_files(selected_checkpoint, model_dir)
        cfg = Config.fromfile(config_path)
        if 'pretrained' in cfg.model:
            cfg.model.pretrained = None
        elif 'init_cfg' in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, weights_path, map_location='cpu')
        
        if model_source == "Custom models":
            classes = cfg.checkpoint_config.meta.CLASSES
            self.selected_model_name = cfg.pretrained_model
            self.checkpoint_name = "custom"
            self.dataset_name = "custom"
            if "segm" in cfg.evaluation.metric:
                obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in classes]
            else:
                obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in classes]
        elif model_source == "Pretrained models":
            dataset_class_name = cfg.dataset_type
            classes = str_to_class(dataset_class_name).CLASSES
            if self.task_type == "object detection":
                obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in classes]
            elif self.task_type == "instance segmentation":
                obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in classes]
            if self.gui is not None:
                self.selected_model_name = list(self.gui.get_model_info().keys())[0]
                checkpoint_info = self.gui.get_checkpoint_info()
                self.checkpoint_name = checkpoint_info["Name"]
                self.dataset_name = checkpoint_info["Dataset"]
            else:
                self.selected_model_name = selected_model_name
                self.checkpoint_name = selected_checkpoint["Name"]
                self.dataset_name = dataset_name

        model.CLASSES = classes
        model.cfg = cfg  # save the config in the model for convenience
        model.to(device)
        model.eval()
        model = revert_sync_batchnorm(model)
        self.model = model
        self.class_names = classes

        self._model_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(obj_classes))
        self._get_confidence_tag_meta()
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        try:
            return self.class_names  # e.g. ["cat", "dog", ...]
        except AttributeError as e:
            exc_message = get_message_from_exception(e)
            raise Exception(
                f"{exc_message}. "
                "You are probably trying to serve model trained outside the Supervisely. "
                "But this app supports custom checkpoints only for models trained in Supervisely via corresponding training app"
            )

    def get_info(self) -> dict:
        info = super().get_info()
        info["task type"] = self.task_type
        info["model_name"] = self.selected_model_name
        info["checkpoint_name"] = self.checkpoint_name
        info["pretrained_on_dataset"] = self.dataset_name
        info["device"] = self.device
        return info

    def get_models(self, add_links=False):
        tasks = ['object detection', 'instance segmentation']
        model_config = {}
        for task_type in tasks:
            model_config[task_type] = {}
            if task_type == 'object detection':
                models_meta_path = os.path.join(root_source_path, "models", "detection_meta.json")
            elif task_type == 'instance segmentation':
                models_meta_path = os.path.join(root_source_path, "models", "instance_segmentation_meta.json")
            model_yamls = sly.json.load_json_file(models_meta_path)
            
            for model_meta in model_yamls:
                mmdet_ver = pkg_resources.get_distribution("mmdet").version
                model_yml_url = f"https://github.com/open-mmlab/mmdetection/tree/v{mmdet_ver}/configs/{model_meta['yml_file']}" 
                model_yml_local = os.path.join(configs_dir, model_meta['yml_file'])
                with open(model_yml_local, "r") as stream:
                    model_info = yaml.safe_load(stream)
                    model_config[task_type][model_meta["model_name"]] = {}
                    model_config[task_type][model_meta["model_name"]]["checkpoints"] = []
                    model_config[task_type][model_meta["model_name"]]["paper_from"] = model_meta["paper_from"]
                    model_config[task_type][model_meta["model_name"]]["year"] = model_meta["year"]
                    model_config[task_type][model_meta["model_name"]]["config_url"] = os.path.dirname(model_yml_url)

                    for model in model_info["Models"]:
                        checkpoint_info = OrderedDict()
                        if "exclude" in model_meta.keys():
                            if model_meta["exclude"].endswith("*"):
                                if model["Name"].startswith(model_meta["exclude"][:-1]):
                                    continue
                        # Saved For Training
                        # checkpoint_info["Use semantic inside"] = False
                        # if "semantic" in model_meta.keys():
                        #     if model_meta["semantic"] == "*":
                        #         checkpoint_info["Use semantic inside"] = True
                        #     elif model_meta["semantic"].startswith("*") and model_meta["semantic"].endswith("*"):
                        #         if model_meta["semantic"][1:-1] in model["Name"]:
                        #             checkpoint_info["Use semantic inside"] = True
                        #     elif model_meta["semantic"].startswith("*") and model["Name"].endswith(model_meta["semantic"][1:]):
                        #         checkpoint_info["Use semantic inside"] = True
                        #     elif model_meta["semantic"].endswith("*") and model_meta["semantic"].startswith("!"):
                        #         if not model["Name"].startswith(model_meta["semantic"][1:-1]):
                        #             checkpoint_info["Use semantic inside"] = True
                            
                        checkpoint_info["Name"] = model["Name"]
                        checkpoint_info["Method"] = model["In Collection"]
                        try:
                            checkpoint_info["Inference Time (ms/im)"] = model["Metadata"]["inference time (ms/im)"][0]["value"]
                        except KeyError:
                            checkpoint_info["Inference Time (ms/im)"] = "-"
                        try:
                            checkpoint_info["Input Size (H, W)"] = model["Metadata"]["inference time (ms/im)"][0]["resolution"]
                        except KeyError:
                            checkpoint_info["Input Size (H, W)"] = "-"
                        try:
                            checkpoint_info["LR scheduler (epochs)"] = model["Metadata"]["Epochs"]
                        except KeyError:
                            checkpoint_info["LR scheduler (epochs)"] = "-"
                        try:
                            checkpoint_info["Memory (Training, GB)"] = model["Metadata"]["Training Memory (GB)"]
                        except KeyError:
                            checkpoint_info["Memory (Training, GB)"] = "-"
                        for result in model["Results"]:
                            if (task_type == "object detection" and result["Task"] == "Object Detection") or \
                            (task_type == "instance segmentation" and result["Task"] == "Instance Segmentation"):
                                checkpoint_info["Dataset"] = result["Dataset"]
                                for metric_name, metric_val in result["Metrics"].items():
                                    checkpoint_info[metric_name] = metric_val
                        try:
                            weights_file = model["Weights"]
                        except KeyError as e:
                            sly.logger.info(f'Weights not found. Model: {model_meta["model_name"]}, checkpoint: {checkpoint_info["Name"]}')
                            continue
                        if add_links:
                            checkpoint_info["config_file"] = model["Config"]
                            checkpoint_info["weights_file"] = weights_file
                        model_config[task_type][model_meta["model_name"]]["checkpoints"].append(checkpoint_info)
        return model_config

    def download_pretrained_files(self, selected_model: Dict[str, str], model_dir: str):
        gui: MMDetectionGUI
        task_type = self.gui.get_task_type()
        models = self.get_models(add_links=True)[task_type]
        if self.gui is not None:
            model_name = list(self.gui.get_model_info().keys())[0]
        else:
            # for local debug without GUI only
            model_name = selected_model_name
        full_model_info = selected_model
        for model_info in models[model_name]["checkpoints"]:
            if model_info["Name"] == selected_model["Name"]:
                full_model_info = model_info
        weights_ext = sly.fs.get_file_ext(full_model_info["weights_file"])
        config_ext = sly.fs.get_file_ext(full_model_info["config_file"])
        weights_dst_path = os.path.join(model_dir, f"{selected_model['Name']}{weights_ext}")
        if not sly.fs.file_exists(weights_dst_path):
            self.download(
                src_path=full_model_info["weights_file"], 
                dst_path=weights_dst_path
            )
        config_path = self.download(
            src_path=full_model_info["config_file"], 
            dst_path=os.path.join(model_dir, f"config{config_ext}")
        )
        
        return weights_dst_path, config_path

    def download_custom_files(self, custom_link: str, model_dir: str):
        weight_filename = os.path.basename(custom_link)
        weights_dst_path = os.path.join(model_dir, weight_filename)
        if not sly.fs.file_exists(weights_dst_path):
            self.download(
                src_path=custom_link,
                dst_path=weights_dst_path,
            )
        config_path = self.download(
            src_path=os.path.join(os.path.dirname(custom_link), 'config.py'),
            dst_path=os.path.join(model_dir, 'config.py'),
        )
        
        return weights_dst_path, config_path

    def initialize_gui(self) -> None:
        models = self.get_models()
        for task_type in ["object detection", "instance segmentation"]:
            for model_group in models[task_type].keys():
                models[task_type][model_group]["checkpoints"] = self._preprocess_models_list(
                    models[task_type][model_group]["checkpoints"]
                )
        self._gui = MMDetectionGUI(
            models,
            self.api,
            support_pretrained_models=True,
            support_custom_models=True,
            custom_model_link_type="file",
        )

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[Union[PredictionBBox, PredictionMask]]:

        result = inference_detector(self.model, image_path)
        torch.cuda.empty_cache()

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
        
        predictions = []
        if segm_result is None:
            for bboxes, class_name in zip(bbox_result, self.get_classes()):
                for bbox in bboxes:
                    top, left, bottom, right, score = int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2]), float(bbox[4])
                    if "confidence_thresh" in settings.keys() and score < settings["confidence_thresh"]:
                        continue
                    predictions.append(
                        PredictionBBox(
                            class_name=class_name, 
                            bbox_tlbr=[top, left, bottom, right], 
                            score=score
                        )
                    )
        else:
            for bboxes, masks, class_name in zip(bbox_result, segm_result, self.get_classes()):
                assert len(bbox_result) == len(segm_result)
                for bbox, mask in zip(bboxes, masks):
                    score = float(bbox[4])
                    if "confidence_thresh" in settings.keys() and score < settings["confidence_thresh"]:
                        continue
                    if not mask.any():
                        continue
                    predictions.append(
                        PredictionMask(
                            class_name=class_name,
                            mask=mask,
                            score=score,
                        )
                    )

        return predictions


if sly.is_production():
    sly.logger.info("Script arguments", extra={
        "context.teamId": sly.env.team_id(),
        "context.workspaceId": sly.env.workspace_id(),
    })

custom_settings_path = os.path.join(app_source_path, "custom_settings.yml")

m = MMDetectionModel(
    use_gui=True, 
    custom_inference_settings=custom_settings_path
)

if sly.is_production() or use_gui_for_local_debug is True:
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    # for local development and debugging without GUI
    task_type = 'object detection'
    models = m.get_models(add_links=True)[task_type]
    selected_model_name = "TOOD"
    dataset_name = "COCO"
    selected_checkpoint = models[selected_model_name]["checkpoints"][0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, m.custom_inference_settings_dict)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print(f"predictions and visualization have been saved: {vis_path}")

