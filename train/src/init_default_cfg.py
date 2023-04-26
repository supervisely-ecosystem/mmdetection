import sly_globals as g
def init_default_cfg_args(state):
    state["epochs"] = 12
    state["valInterval"] = 12
    state["batchSizePerGPU"] = 2
    state["workersPerGPU"] = 2
    state["optimizer"] = "SGD"
    state["lr"] = 0.001
    state["weightDecay"] = 0
    state["gradClipEnabled"] = True
    state["maxNorm"] = 10
    state["lrPolicy"] = "Fixed"
    state["availableLrPolicy"] = ["Fixed", "Step", "Exp", "Poly", "Inv", "CosineAnnealing", "FlatCosineAnnealing",
                                 "CosineRestart", "Cyclic", "OneCycle"]
    state["lr_step"] = ""
    state["useWarmup"] = False
    state["warmup"] = "constant"
    state["warmupIters"] = 0
    state["warmupRatio"] = 0.1
    state["schedulerByEpochs"] = True
    state["warmupByEpoch"] = False
    state["minLREnabled"] = False
    state["minLR"] = None
    state["minLRRatio"] = None
    state["power"] = 1
    state["momentum"] = 0.9
    state["beta1"] = 0.9
    state["beta2"] = 0.999
    state["input_size"] = {
        "value": {
            "width": 256,
            "height": 256,
            "proportional": False
        },
        "options": {
            "proportions": {
              "width": 100,
              "height": 100
            },
            "min": 64
        }
    }


def rewrite_default_cfg_args(cfg, state):
    params = []
    if hasattr(cfg.data, "samples_per_gpu"):
        params.extend([{
            "field": "state.batchSizePerGPU",
            "payload": cfg.data.samples_per_gpu
        }])
    if hasattr(cfg.data, "workers_per_gpu"):
        params.extend([{
            "field": "state.workersPerGPU",
            "payload": cfg.data.workers_per_gpu
        }])
    if hasattr(cfg, "runner") and hasattr(cfg.runner, "max_epochs"):
        params.extend([{
            "field": "state.epochs",
            "payload": cfg.runner.max_epochs
        }])
    if hasattr(cfg.evaluation, "interval") and cfg.evaluation.interval > state["valInterval"]:
        params.extend([{
            "field": "state.valInterval",
            "payload": cfg.evaluation.interval
        }])
    for pipeline_step in cfg.data.test["pipeline"]:
        if pipeline_step["type"] == "MultiScaleFlipAug":
            img_scale = None
            if "img_scale" in pipeline_step.keys():
                img_scale = pipeline_step["img_scale"]
            else:
                train_data = cfg.data.train
                if "pipeline" not in train_data.keys() and "pipeline" in train_data.dataset.keys():
                    train_data = train_data.dataset
                for train_step in train_data["pipeline"]:
                    if train_step["type"] == "Resize":
                        img_scale = train_step["img_scale"]
            assert img_scale is not None

            params.extend([{
                "field": "state.input_size.value.height",
                "payload": img_scale[0]
            },{
                "field": "state.input_size.value.width",
                "payload": img_scale[1]
            },{
                "field": "state.input_size.options.proportions.height",
                "payload": 100
            },{
                "field": "state.input_size.options.proportions.width",
                "payload": 100 * (img_scale[1] / img_scale[0])
            }])
    if hasattr(cfg.optimizer, "type"):
        params.extend([{
            "field": "state.optimizer",
            "payload": cfg.optimizer.type
        }])
    if hasattr(cfg.optimizer, "lr"):
        params.extend([{
            "field": "state.lr",
            "payload": cfg.optimizer.lr
        }])
    if hasattr(cfg.optimizer, "weight_decay"):
        params.extend([{
            "field": "state.weightDecay",
            "payload": cfg.optimizer.weight_decay
        }])
    if hasattr(cfg.optimizer, "momentum"):
        params.extend([{
            "field": "state.momentum",
            "payload": cfg.optimizer.momentum
        }])
    if hasattr(cfg.optimizer, "betas"):
        params.extend([{
            "field": "state.beta1",
            "payload": cfg.optimizer.betas[0]
        },{
            "field": "state.beta2",
            "payload": cfg.optimizer.betas[1]
        }])
    if hasattr(cfg.optimizer_config.grad_clip, "max_norm"):
        params.extend([{
            "field": "state.maxNorm",
            "payload": cfg.optimizer_config.grad_clip.max_norm
        },{
            "field": "state.gradClipEnabled",
            "payload": True
        }])

    # take lr scheduler params
    if hasattr(cfg, "lr_config"):
        # warmup
        if hasattr(cfg.lr_config, "warmup"):
            warmup = cfg.lr_config.warmup
            params.extend([{
                "field": "state.useWarmup",
                "payload": warmup is not None
            },{
                "field": "state.warmup",
                "payload": warmup
            }])
        
        if hasattr(cfg.lr_config, "warmup_iters"):
            warmup_iters = cfg.lr_config.warmup_iters
            # warmup iters no more than half of all data length
            if warmup_iters > g.project_info.items_count * 0.5 // state["batchSizePerGPU"]:
                warmup_iters = g.project_info.items_count * 0.5 // state["batchSizePerGPU"]
            params.extend([{
                "field": "state.warmupIters",
                "payload": warmup_iters
            }])
        if hasattr(cfg.lr_config, "warmup_ratio"):
            params.extend([{
                "field": "state.warmupRatio",
                "payload": cfg.lr_config.warmup_ratio
            }])
        if hasattr(cfg.lr_config, "warmup_by_epoch"):
            params.extend([{
                "field": "state.warmupByEpochs",
                "payload": cfg.lr_config.warmup_by_epoch
            }])
        # policy
        if hasattr(cfg.lr_config, "policy"):
            policy = cfg.lr_config.policy.capitalize()
            if policy in state["availableLrPolicy"]:
                params.extend([{
                    "field": "state.lrPolicy",
                    "payload": policy
                }])
            else:
                return params
        if hasattr(cfg.lr_config, "step"):
            params.extend([{
                "field": "state.lr_step",
                "payload": ",".join([str(step) for step in cfg.lr_config.step])
            }])
        
        if hasattr(cfg.lr_config, "by_epoch"):
            params.extend([{
                "field": "state.schedulerByEpochs",
                "payload": cfg.lr_config.by_epoch
            }])
        if hasattr(cfg.lr_config, "min_lr"):
            params.extend([{
                "field": "state.minLREnabled",
                "payload": True
            },{
                "field": "state.minLR",
                "payload": cfg.lr_config.min_lr
            }])
        if hasattr(cfg.lr_config, "power"):
            params.extend([{
                "field": "state.power",
                "payload": cfg.lr_config.power
            }])

    return params