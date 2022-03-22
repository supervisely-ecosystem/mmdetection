import architectures
import augs
import splits
import sly_globals as g
from mmcv import ConfigDict
from mmdet.apis import set_random_seed
import os.path as osp

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
        if not hasattr(cfg, "optimizer_config"):
            cfg.optimizer_config = ConfigDict(
                grad_clip=ConfigDict(max_norm=state["maxNorm"], norm_type=2)
            )
        else:
            cfg.optimizer_config.grad_clip=ConfigDict(max_norm=state["maxNorm"], norm_type=2)
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
    if cfg.pretrained_model == "YOLOX":
        cfg.data.train.dataset.pipeline.append(dict(type='SlyImgAugs', config_path=augs.augs_config_path))
        train_steps_to_remove = ["RandomAffine", "YOLOXHSVRandomAug"]
        train_pipeline = []
        for config_step in cfg.data.train.pipeline:
            if config_step["type"] in train_steps_to_remove:
                continue
            train_pipeline.append(config_step)
        cfg.data.train.pipeline = train_pipeline
        return
    train_steps_to_remove = ["RandomShift", "PhotoMetricDistortion"]
    train_pipeline = []
    for config_step in cfg.data.train.pipeline:
        if config_step["type"] in train_steps_to_remove:
            continue
        elif config_step["type"] == "LoadAnnotations":
            train_pipeline.append(config_step)
            train_pipeline.append(dict(type='SlyImgAugs', config_path=augs.augs_config_path))
            continue
        train_pipeline.append(config_step)
    cfg.data.train.pipeline = train_pipeline


def init_cfg_splits(cfg, classes, palette, task):
    cfg.dataset_type = "SuperviselyDataset"
    cfg.data_root = g.project_dir

    train_dataset = cfg.data.train
    val_dataset = cfg.data.val
    test_dataset = cfg.data.test

    if cfg.data.train.type == "RepeatDataset":
        cfg.data.train = cfg.data.train.dataset
        train_dataset = cfg.data.train
    elif cfg.data.train.type == "MultiImageMixDataset":
        train_dataset = cfg.data.train.dataset

    train_dataset.data_root = cfg.data_root
    train_dataset.type = cfg.dataset_type
    train_dataset.ann_file = splits.train_set_path
    train_dataset.img_prefix = ''
    train_dataset.test_mode = False
    train_dataset.classes = classes
    if cfg.with_semantic_masks:
        train_dataset.seg_prefix = osp.join(cfg.work_dir, "seg")
    if hasattr(train_dataset, "times"):
        delattr(train_dataset, "times")
    if hasattr(train_dataset, "dataset"):
        delattr(train_dataset, "dataset")

    val_dataset.data_root = cfg.data_root
    val_dataset.type = cfg.dataset_type
    val_dataset.ann_file = splits.val_set_path
    val_dataset.img_prefix = ''
    val_dataset.test_mode = False
    val_dataset.classes = classes
    if cfg.with_semantic_masks:
        val_dataset.seg_prefix = osp.join(cfg.work_dir, "seg")
    # TODO: decside what to do with this
    # val_dataset.samples_per_gpu = 2
    
    test_dataset.data_root = cfg.data_root
    test_dataset.type = cfg.dataset_type
    test_dataset.ann_file = splits.val_set_path
    test_dataset.img_prefix = ''
    test_dataset.test_mode = True
    test_dataset.classes = classes


def init_cfg_training(cfg, state):
    cfg.seed = 0
    set_random_seed(cfg.seed, deterministic=False)

    cfg.data.samples_per_gpu = state["batchSizePerGPU"]
    cfg.data.workers_per_gpu = state["workersPerGPU"]
    cfg.data.persistent_workers = True

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
        steps = [int(step) for step in state["lr_step"].split(",")]
        assert len(steps)
        lr_config["step"] = steps
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


def init_model(cfg, classes, state):
    # modify num classes of the model in box head
    if state["task"] == "detection" or (state["task"] == "instance_segmentation" and cfg.pretrained_model != "SOLO"):
        if hasattr(cfg.model, "roi_head"):
            if hasattr(cfg.model.roi_head, "bbox_head") and not isinstance(cfg.model.roi_head.bbox_head, list):
                cfg.model.roi_head.bbox_head.num_classes = len(classes)
                if cfg.model.roi_head.bbox_head.loss_bbox["type"] == "SmoothL1Loss":
                    cfg.model.roi_head.bbox_head.loss_bbox = ConfigDict(type="MSELoss", loss_weight=cfg.model.roi_head.bbox_head.loss_bbox["loss_weight"])
            elif hasattr(cfg.model.roi_head, "bbox_head") and isinstance(cfg.model.roi_head.bbox_head, list):
                for i in range(len(cfg.model.roi_head.bbox_head)):
                    cfg.model.roi_head.bbox_head[i].num_classes = len(classes)
                    if cfg.model.roi_head.bbox_head[i].loss_bbox["type"] == "SmoothL1Loss":
                        cfg.model.roi_head.bbox_head[i].loss_bbox = ConfigDict(type="MSELoss", loss_weight=cfg.model.roi_head.bbox_head[i].loss_bbox["loss_weight"])
            else:
                raise ValueError("No bbox head in roi head")
                
        elif hasattr(cfg.model, "bbox_head") and not isinstance(cfg.model.bbox_head, list):
            cfg.model.bbox_head.num_classes = len(classes)
        elif hasattr(cfg.model, "bbox_head") and isinstance(cfg.model.bbox_head, list):
            for i in range(len(cfg.model.bbox_head)):
                cfg.model.bbox_head[i].num_classes = len(classes)
        else:
            raise ValueError("No bbox head.")

        if hasattr(cfg.model, "rpn_head") and hasattr(cfg.model.rpn_head, "loss_bbox"):
            if cfg.model.rpn_head.loss_bbox["type"] == "SmoothL1Loss":
                cfg.model.rpn_head.loss_bbox = ConfigDict(type="MSELoss", loss_weight=cfg.model.rpn_head.loss_bbox["loss_weight"])

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

            if hasattr(cfg.model.roi_head, "semantic_head"):
                # TODO: check len classes in semantic head
                cfg.model.roi_head.semantic_head.num_classes = len(classes) +1
                cfg.model.roi_head.semantic_head.type = 'SlyFusedSemanticHead'

        if hasattr(cfg.model, "segm_head"):
            cfg.model.segm_head.num_classes = len(classes)

def init_cfg(state, classes, palette):
    cfg = architectures.cfg

    # class_weights = init_class_weights(state, classes)
    init_model(cfg, classes, state)
    init_cfg_optimizer(cfg, state)
    init_cfg_training(cfg, state)
    init_cfg_splits(cfg, classes, palette, state["task"])
    init_cfg_pipelines(cfg)
    init_cfg_eval(cfg, state)
    init_cfg_checkpoint(cfg, state, classes, palette)
    init_cfg_lr(cfg, state)

    return cfg