import supervisely as sly
import sly_globals as g
import splits

def init_general(state):
    state["gpusId"] = 0
    state["logConfigInterval"] = 5

def init_checkpoints(state):
    state["checkpointInterval"] = 12
    state["maxKeepCkptsEnabled"] = True
    state["maxKeepCkpts"] = 1
    state["saveLast"] = False
    state["saveBest"] = True

def init_optimizer(state):
    state["nesterov"] = False
    state["amsgrad"] = False
    state["momentumDecay"] = 0.004

def init_losses(data, state):
    state["useClassWeights"] = False
    state["classWeights"] = ""
    data["classesList"] = [class_obj["title"] for class_obj in g.project_meta.obj_classes.to_json()]


def init_lr_scheduler(data, state):
    # LR scheduler params
    data["fullPolicyNames"] = ["Constant LR", "Step LR", "Exponential LR", "Polynomial LR Decay",
                               "Inverse Square Root LR", "Cosine Annealing LR", "Flat + Cosine Annealing LR",
                               "Cosine Annealing with Restarts", "Cyclic LR", "OneCycle LR"]

    state["gamma"] = 0.1
    state["startPercent"] = 0.75
    state["periods"] = ""
    state["restartWeights"] = ""
    state["highestLRRatio"] = 10
    state["lowestLRRatio"] = 1e-3
    state["cyclicTimes"] = 10
    state["stepRatioUp"] = 0.4
    state["annealStrategy"] = "cos"
    # state["cyclicGamma"] = 1
    state["totalStepsEnabled"] = False
    state["totalSteps"] = None
    state["maxLR"] = ""
    state["pctStart"] = 0.3
    state["divFactor"] = 25
    state["finalDivFactor"] = 1e4
    state["threePhase"] = False

def init(data, state):
    init_general(state)
    init_checkpoints(state)
    init_optimizer(state)
    # init_losses(data, state)
    init_lr_scheduler(data, state)

    state["currentTab"] = "general"
    state["collapsedWarmup"] = True
    state["collapsedHyperparams"] = True
    state["disabledHyperparams"] = True
    state["doneHyperparams"] = False


def restart(data, state):
    data["doneHyperparams"] = False


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.doneHyperparams", "payload": True},
        {"field": "state.collapsedMonitoring", "payload": False},
        {"field": "state.disabledMonitoring", "payload": False},
        {"field": "state.activeStep", "payload": 8},
    ]
    if state["batchSizePerGPU"] > len(splits.train_set):
        fields.append({"field": "state.batchSizePerGPU", "payload": len(splits.train_set)})
        g.my_app.show_modal_window(
            f"Specified batch size is more than train split length. Batch size will be equal to length of train split ({len(splits.train_set)})."
        )
    g.api.app.set_fields(g.task_id, fields)