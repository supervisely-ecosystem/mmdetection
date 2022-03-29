import supervisely as sly
import sly_globals as g
import architectures

def init(data, state):
    # TODO: написать на что влияет выбор таски и в чем их отличия (для нубов)
    state["task"] = "detection"
    state["collapsedTask"] = True
    state["disabledTask"] = True
    state["splitInProgress"] = False
    data["doneTask"] = False


def restart(data, state):
    data["doneTask"] = False


def check_labels(task):
    if task == "detection":
        return
    # check for instance seg task
    stats = g.api.project.get_stats(g.project_id)
    seg_classes = [obj["title"] for obj in g.project_meta.obj_classes.to_json() if obj["shape"] in ["bitmap", "polygon"]]
    
    for item in stats["images"]["objectClasses"]:
        if item["objectClass"]["name"] in seg_classes and item["total"] == 0:
            seg_classes.remove(item["objectClass"]["name"])
    if len(seg_classes) == 0:
        g.my_app.show_modal_window(
            "Segmentation classes (bitmap, polygon) not found in project. Instance segmentation task is not available."
        )
        raise ValueError()


@g.my_app.callback("select_task")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def select_task(api: sly.Api, task_id, context, state, app_logger):
    g.api.app.set_field(g.task_id, "state.splitInProgress", True)
    try:
        check_labels(state["task"])
    except ValueError as e:
        return
    # TODO: add loading text while update architectures is performing
    architectures.reload_task(state["task"])
    fields = [
        {"field": "state.collapsedModels", "payload": False},
        {"field": "state.disabledModels", "payload": False},
        {"field": "state.splitInProgress", "payload": False},
        {"field": "data.doneTask", "payload": True},
        {"field": "state.activeStep", "payload": 3},
    ]
    g.api.app.set_fields(g.task_id, fields)
    

    
    