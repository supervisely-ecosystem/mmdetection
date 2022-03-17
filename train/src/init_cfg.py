import architectures
import augs
import splits
import os
import sly_globals as g
from mmcv import ConfigDict
from mmdet.apis import set_random_seed
import numpy as np
import torch
import random

def init_class_weights(state, classes):
    if state["useClassWeights"]:
        return [float(weight) for weight in state["classWeights"].split(",")]
    else:
        return [1] * len(classes)

def init_cfg_optimizer(cfg, state):
    cfg.optimizer.type = state["optimizer"]
    cfg.optimizer.lr = state["lr"]
    cfg.optimizer.weight_decay = state["weightDecay"]
    if state["gradClipEnabled"]:
        cfg.optimizer_config = ConfigDict(
            grad_clip=ConfigDict(max_norm=state["maxNorm"], norm_type=2)
        )
    if hasattr(cfg.optimizer, "eps"):
        delattr(cfg.optimizer, "eps")

    if state["optimizer"] == "SGD":
        if hasattr(cfg.optimizer, "betas"):
            delattr(cfg.optimizer, "betas")
        cfg.optimizer.momentum = state["momentum"]
        cfg.optimizer.nesterov = state["nesterov"]
    elif state["optimizer"] in ["Adam", "Adamax", "AdamW", "NAdam", "RAdam"]:
        if hasattr(cfg.optimizer, "momentum"):
            delattr(cfg.optimizer, "momentum")
        if hasattr(cfg.optimizer, "nesterov"):
            delattr(cfg.optimizer, "nesterov")
        cfg.optimizer.betas = (state["beta1"], state["beta2"])
        if state["optimizer"] in ["Adam", "AdamW"]:
            cfg.optimizer.amsgrad = state["amsgrad"]
        if state["optimizer"] == "NAdam":
            cfg.optimizer.momentum_decay = state["momentumDecay"]

def init_cfg_pipelines(cfg):
    pipeline_steps_to_remove = ["RandomShift", "PhotoMetricDistortion"]
    train_pipeline = []
    for config_step in cfg.data.train.pipeline:
        if config_step["type"] in pipeline_steps_to_remove:
            continue
        elif config_step["type"] == "LoadAnnotations":
            train_pipeline.append(config_step)
            train_pipeline.append(dict(type='SlyImgAugs', config_path=augs.augs_config_path))
            continue
        elif config_step["type"] == "LoadImageFromFile" and "to_float32" in config_step.keys():
            if config_step["to_float32"]:
                train_pipeline.append(dict(type="LoadImageFromFile"))
            continue
        train_pipeline.append(config_step)
    
    cfg.data.train.pipeline = train_pipeline


def init_cfg_splits(cfg, classes, palette, task):
        cfg.data.train.type = cfg.dataset_type
        cfg.data.train.data_root = cfg.data_root
        cfg.data.train.ann_file = splits.train_set_path
        cfg.data.train.img_prefix = ''
        cfg.data.train.seg_prefix = None
        cfg.data.train.proposal_file = None
        cfg.data.train.test_mode = False
        cfg.data.train.classes = classes
        if hasattr(cfg.data.train, "times"):
            delattr(cfg.data.train, "times")
        if hasattr(cfg.data.train, "dataset"):
            delattr(cfg.data.train, "dataset")

        cfg.data.val.type = cfg.dataset_type
        cfg.data.val.data_root = cfg.data_root
        cfg.data.val.ann_file = splits.val_set_path
        cfg.data.val.img_prefix = ''
        cfg.data.val.seg_prefix = None
        cfg.data.val.proposal_file = None
        cfg.data.val.test_mode = False
        cfg.data.val.classes = classes
        cfg.data.val.samples_per_gpu = 2

        cfg.data.test.type = cfg.dataset_type
        cfg.data.test.data_root = cfg.data_root
        cfg.data.test.ann_file = splits.val_set_path
        cfg.data.test.img_prefix = ''
        cfg.data.test.seg_prefix = None
        cfg.data.test.proposal_file = None
        cfg.data.test.test_mode = True
        cfg.data.test.classes = classes


def init_cfg_training(cfg, state):
    cfg.dataset_type = 'SuperviselyDataset'
    cfg.data_root = g.project_dir

    cfg.seed = 0
    set_random_seed(cfg.seed, deterministic=False)

    def worker_init_fn(worker_id):
        np.random.seed(cfg.seed + worker_id)
        random.seed(cfg.seed + worker_id)
        torch.manual_seed(cfg.seed + worker_id)
    cfg.data.samples_per_gpu = state["batchSizePerGPU"]
    cfg.data.workers_per_gpu = state["workersPerGPU"]
    cfg.data.persistent_workers = True
    cfg.data.worker_init_fn = worker_init_fn
    cfg.data.prefetch_factor = 1

    # TODO: sync with state["gpusId"] if it will be needed
    cfg.gpu_ids = range(1)
    cfg.load_from = g.local_weights_path

    cfg.work_dir = g.my_app.data_dir

    if not hasattr(cfg, "runner"):
        cfg.runner = ConfigDict()
    cfg.runner.type = "EpochBasedRunner"
    cfg.runner.max_epochs = state["epochs"]
    if hasattr(cfg, "total_epochs"):
        cfg.total_epochs = state["epochs"]
    if hasattr(cfg.runner, "max_iters"):
        delattr(cfg.runner, "max_iters")

    cfg.log_config.interval = state["logConfigInterval"]
    cfg.log_config.hooks = [
        dict(type='SuperviselyLoggerHook', by_epoch=False)
    ]

def init_cfg_eval(cfg, state):
    cfg.evaluation.interval = state["valInterval"]
    metrics = ['bbox']
    if state["task"] == "instance_segmentation":
        metrics.append('segm')
    cfg.evaluation.metric = metrics
    cfg.evaluation.save_best = "auto" if state["saveBest"] else None
    cfg.evaluation.rule = "greater"
    cfg.evaluation.out_dir = g.checkpoints_dir
    cfg.evaluation.by_epoch = True
    cfg.evaluation.classwise = True

def init_cfg_checkpoint(cfg, state, classes, palette):
    cfg.checkpoint_config.interval = state["checkpointInterval"]
    cfg.checkpoint_config.by_epoch = True
    cfg.checkpoint_config.max_keep_ckpts = state["maxKeepCkpts"] if state["maxKeepCkptsEnabled"] else None
    cfg.checkpoint_config.save_last = state["saveLast"]
    cfg.checkpoint_config.out_dir = g.checkpoints_dir
    cfg.checkpoint_config.meta = ConfigDict(
        CLASSES=classes,
        PALETTE=palette)

def init_cfg_lr(cfg, state):
    lr_config = ConfigDict(
        policy=state["lrPolicy"],
        by_epoch=state["schedulerByEpochs"],
        warmup=state["warmup"] if state["useWarmup"] else None,
        warmup_by_epoch=state["warmupByEpoch"],
        warmup_iters=state["warmupIters"],
        warmup_ratio=state["warmupRatio"]
    )
    if state["lrPolicy"] == "Step":
        lr_config["lr_step"] = [int(step) for step in state["lr_step"].split(",")]
        lr_config["gamma"] = state["gamma"]
        lr_config["min_lr"] = state["minLR"]
    elif state["lrPolicy"] == "Exp":
        lr_config["gamma"] = state["gamma"]
    elif state["lrPolicy"] == "Poly":
        lr_config["min_lr"] = state["minLR"]
        lr_config["power"] = state["power"]
    elif state["lrPolicy"] == "Inv":
        lr_config["gamma"] = state["gamma"]
        lr_config["power"] = state["power"]
    elif state["lrPolicy"] == "CosineAnnealing":
        lr_config["min_lr"] = state["minLR"] if state["minLREnabled"] else None
        lr_config["min_lr_ratio"] = state["minLRRatio"] if not state["minLREnabled"] else None
    elif state["lrPolicy"] == "FlatCosineAnnealing":
        lr_config["min_lr"] = state["minLR"] if state["minLREnabled"] else None
        lr_config["min_lr_ratio"] = state["minLRRatio"] if not state["minLREnabled"] else None
        lr_config["start_percent"] = state["startPercent"]
    elif state["lrPolicy"] == "CosineRestart":
        lr_config["min_lr"] = state["minLR"] if state["minLREnabled"] else None
        lr_config["min_lr_ratio"] = state["minLRRatio"] if not state["minLREnabled"] else None
        lr_config["periods"] = [int(period) for period in state["periods"].split(",")]
        lr_config["restart_weights"] = [float(weight) for weight in state["restartWeights"].split(",")]
    elif state["lrPolicy"] == "Cyclic":
        lr_config["target_ratio"] = (state["highestLRRatio"], state["lowestLRRatio"])
        lr_config["cyclic_times"] = state["cyclicTimes"]
        lr_config["step_ratio_up"] = state["stepRatioUp"]
        lr_config["anneal_strategy"] = state["annealStrategy"]
        # lr_config["gamma"] = state["cyclicGamma"]
    elif state["lrPolicy"] == "OneCycle":
        lr_config["anneal_strategy"] = state["annealStrategy"]
        lr_config["max_lr"] = [float(maxlr) for maxlr in state["maxLR"].split(",")]
        lr_config["total_steps"] = state["totalSteps"] if state["totalStepsEnabled"] else None
        lr_config["pct_start"] = state["pctStart"]
        lr_config["div_factor"] = state["divFactor"]
        lr_config["final_div_factor"] = state["finalDivFactor"]
        lr_config["three_phase"] = state["threePhase"]
    cfg.lr_config = lr_config

def init_cfg(state, classes, palette):
    cfg = architectures.cfg

    # Since we use ony one GPU, BN is used instead of SyncBN
    # cfg.norm_cfg = dict(type='BN', requires_grad=True)
    # if cfg.pretrained_model not in ["DPT", "SegFormer", "SETR", "Swin Transformer", "Twins", "ViT"]:
    #     cfg.model.backbone.norm_cfg = cfg.norm_cfg

    # class_weights = init_class_weights(state, classes)
    
    # modify num classes of the model in box head
    if state["task"] == "detection" or (state["task"] == "instance_segmentation" and cfg.pretrained_model != "SOLO"):
        if hasattr(cfg.model, "roi_head"):
            if hasattr(cfg.model.roi_head, "bbox_head") and not isinstance(cfg.model.roi_head.bbox_head, list):
                cfg.model.roi_head.bbox_head.num_classes = len(classes)
            elif hasattr(cfg.model.roi_head, "bbox_head") and isinstance(cfg.model.roi_head.bbox_head, list):
                for i in range(len(cfg.model.roi_head.bbox_head)):
                    cfg.model.roi_head.bbox_head[i].num_classes = len(classes)
            else:
                raise ValueError("No bbox head in roi head")
                
        elif hasattr(cfg.model, "bbox_head") and not isinstance(cfg.model.bbox_head, list):
            cfg.model.bbox_head.num_classes = len(classes)
        elif hasattr(cfg.model, "bbox_head") and isinstance(cfg.model.bbox_head, list):
            for i in range(len(cfg.model.bbox_head)):
                cfg.model.bbox_head[i].num_classes = len(classes)
        else:
            raise ValueError("No bbox head.")

    # modify num classes of the model in mask head
    if state["task"] == "instance_segmentation":
        if hasattr(cfg.model, "roi_head"):
            if hasattr(cfg.model.roi_head, "mask_head") and not isinstance(cfg.model.roi_head.mask_head, list):
                cfg.model.roi_head.mask_head.num_classes = len(classes)
            elif hasattr(cfg.model.roi_head, "mask_head") and isinstance(cfg.model.roi_head.mask_head, list):
                for i in range(len(cfg.model.roi_head.mask_head)):
                    cfg.model.roi_head.mask_head[i].num_classes = len(classes)
            else:
                raise ValueError("No mask head in roi head")
        elif hasattr(cfg.model, "mask_head") and not isinstance(cfg.model.mask_head, list):
            cfg.model.mask_head.num_classes = len(classes)
        elif hasattr(cfg.model, "mask_head") and isinstance(cfg.model.mask_head, list):
            for i in range(len(cfg.model.mask_head)):
                cfg.model.mask_head[i].num_classes = len(classes)
        else:
            raise ValueError("No mask head.")

        # other heads
        if hasattr(cfg.model, "roi_head"):
            if hasattr(cfg.model.roi_head, "glbctx_head"):
                cfg.model.roi_head.glbctx_head.num_classes = len(classes)
            
            if hasattr(cfg.model.roi_head, "point_head"):
                cfg.model.roi_head.point_head.num_classes = len(classes)

            if hasattr(cfg.model.roi_head, "mask_iou_head"):
                cfg.model.roi_head.mask_iou_head.num_classes = len(classes)

        if hasattr(cfg.model, "segm_head"):
            cfg.model.segm_head.num_classes = len(classes)
    # init_cfg_optimizer(cfg, state)
    # init_cfg_training(cfg, state)
    cfg.evaluation.interval = 12
    cfg.data_root = g.project_dir
    cfg.load_from = g.local_weights_path
    cfg.work_dir = g.my_app.data_dir
    cfg.log_config.interval = state["logConfigInterval"]
    cfg.log_config.hooks = [
        dict(type='SuperviselyLoggerHook', by_epoch=False)
    ]
    # init_cfg_pipelines(cfg)
    # init_cfg_splits(cfg, classes, palette, state["task"])
    cfg.dataset_type = "CocoDataset"
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.ann_file = splits.train_set_path
    cfg.data.train.img_prefix = ''
    cfg.data.train.test_mode = False
    cfg.data.train.classes = classes
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.ann_file = splits.val_set_path
    cfg.data.val.img_prefix = ''
    cfg.data.val.test_mode = False
    cfg.data.val.classes = classes
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.ann_file = splits.val_set_path
    cfg.data.test.img_prefix = ''
    cfg.data.test.test_mode = True
    cfg.data.test.classes = classes
    cfg.gpu_ids = range(1)
    cfg.seed = 0
    set_random_seed(cfg.seed, deterministic=False)
    # init_cfg_eval(cfg, state)
    # init_cfg_checkpoint(cfg, state, classes, palette)
    # init_cfg_lr(cfg, state)

    return cfg