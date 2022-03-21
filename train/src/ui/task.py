import supervisely as sly
import sly_globals as g
import architectures

def init(data, state):
    # TODO: написать на что влияет выбор таски и в чем их отличия (для нубов)
    state["task"] = "detection"
    state["collapsedTask"] = True
    state["disabledTask"] = True
    data["doneTask"] = False


def restart(data, state):
    data["doneTask"] = False


@g.my_app.callback("select_task")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def select_task(api: sly.Api, task_id, context, state, app_logger):
    # TODO: check if task is "instance_segmentation" but labels are only bboxes
    # TODO: add loading text while update architectures is performing
    architectures.reload_task(state["task"])
    fields = [
        {"field": "state.collapsedClasses", "payload": False},
        {"field": "state.disabledClasses", "payload": False},
        {"field": "data.doneTask", "payload": True},
        {"field": "state.activeStep", "payload": 3},
    ]
    g.api.app.set_fields(g.task_id, fields)
    