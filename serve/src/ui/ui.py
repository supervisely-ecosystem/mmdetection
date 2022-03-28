import supervisely as sly
import sly_globals as g
import task
import architectures


@sly.timeit
def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None
    task.init(data, state)
    architectures.init(data, state)


@g.my_app.callback("restart")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def restart(api: sly.Api, task_id, context, state, app_logger):
    restart_from_step = state["restartFrom"]
    data = {}
    state = {}

    task.restart(data, state)
    architectures.init(data, state)

    fields = [
        {"field": "data", "payload": data, "append": True, "recursive": False},
        {"field": "state", "payload": state, "append": True, "recursive": False},
        {"field": "state.restartFrom", "payload": None},
        {"field": "state.activeStep", "payload": restart_from_step},
        {"field": "data.scrollIntoView", "payload": f"step{restart_from_step}"},
    ]
    g.api.app.set_fields(g.task_id, fields)