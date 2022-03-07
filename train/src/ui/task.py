import supervisely as sly
import sly_globals as g

def init(data, state):
    state["task"] = "detection"
    state["collapsedTask"] = True
    state["disabledTask"] = True
    data["doneTask"] = False

@g.my_app.callback("select_task")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def select_task(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "state.task", "payload": state["task"]},
        {"field": "state.collapsedClasses", "payload": False},
        {"field": "state.disabledClasses", "payload": False},
        {"field": "data.doneTask", "payload": True},
        {"field": "state.activeStep", "payload": 3},
    ]
    g.api.app.set_fields(g.task_id, fields)
    