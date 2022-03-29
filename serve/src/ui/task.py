import supervisely as sly
import sly_globals as g
import architectures

def init(data, state):
    # TODO: написать на что влияет выбор таски и в чем их отличия (для нубов)
    state["task"] = "detection"
    state["collapsedTask"] = False
    state["disabledTask"] = False
    state["modelsUpdating"] = False
    data["doneTask"] = False
    state["deployed"] = False


def restart(data, state):
    data["doneTask"] = False


@g.my_app.callback("select_task")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def select_task(api: sly.Api, task_id, context, state, app_logger):
    g.api.app.set_field(g.TASK_ID, "state.modelsUpdating", True)
    architectures.reload_task(state["task"])
    fields = [
        {"field": "state.collapsedModels", "payload": False},
        {"field": "state.disabledModels", "payload": False},
        {"field": "state.modelsUpdating", "payload": False},
        {"field": "data.doneTask", "payload": True},
        {"field": "state.activeStep", "payload": 2},
    ]
    g.api.app.set_fields(g.TASK_ID, fields)
    