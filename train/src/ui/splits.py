import supervisely as sly
import sly_globals as g
import os
from input_project import get_image_info_from_cache

train_set = None
val_set = None

train_set_path = os.path.join(g.my_app.data_dir, "train.json")
val_set_path = os.path.join(g.my_app.data_dir, "val.json")

def init(project_info, project_meta: sly.ProjectMeta, data, state):
    data["randomSplit"] = [
        {"name": "train", "type": "success"},
        {"name": "val", "type": "primary"},
        {"name": "total", "type": "gray"},
    ]
    data["totalImagesCount"] = project_info.items_count

    train_percent = 80
    train_count = int(project_info.items_count / 100 * train_percent)
    state["randomSplit"] = {
        "count": {
            "total": project_info.items_count,
            "train": train_count,
            "val": project_info.items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    state["splitMethod"] = "random"

    state["trainTagName"] = ""
    if project_meta.tag_metas.get("train") is not None:
        state["trainTagName"] = "train"
    state["valTagName"] = ""
    if project_meta.tag_metas.get("val") is not None:
        state["valTagName"] = "val"

    state["trainDatasets"] = []
    state["valDatasets"] = []
    state["untaggedImages"] = "train"
    state["splitInProgress"] = False
    state["trainImagesCount"] = None
    state["valImagesCount"] = None
    state["fullTrainImagesCount"] = None
    state["fullValImagesCount"] = None
    data["done2"] = False
    state["collapsed2"] = True
    state["disabled2"] = True


def get_train_val_sets(project_dir, state):
    split_method = state["splitMethod"]
    if split_method == "random":
        train_count = state["randomSplit"]["count"]["train"]
        val_count = state["randomSplit"]["count"]["val"]
        train_set, val_set = sly.Project.get_train_val_splits_by_count(project_dir, train_count, val_count)
        return train_set, val_set
    elif split_method == "tags":
        train_tag_name = state["trainTagName"]
        val_tag_name = state["valTagName"]
        add_untagged_to = state["untaggedImages"]
        train_set, val_set = sly.Project.get_train_val_splits_by_tag(project_dir, train_tag_name, val_tag_name,
                                                                     add_untagged_to)
        return train_set, val_set
    elif split_method == "datasets":
        train_datasets = state["trainDatasets"]
        val_datasets = state["valDatasets"]
        train_set, val_set = sly.Project.get_train_val_splits_by_dataset(project_dir, train_datasets, val_datasets)
        return train_set, val_set
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")


def remove_empty_items(split):
    items_to_include = []
    for item in split:
        img_info = get_image_info_from_cache(item.dataset_name, item.name)
        if img_info.labels_count > 0:
            items_to_include.append(item)
    return items_to_include


@g.my_app.callback("create_splits")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def create_splits(api: sly.Api, task_id, context, state, app_logger):
    step_done = False
    train_len = None
    val_len = None
    global train_set, val_set
    try:
        api.task.set_field(task_id, "state.splitInProgress", True)
        
        train_set, val_set = get_train_val_sets(g.project_dir, state)
        sly.logger.info(f"Train set: {len(train_set)} images")
        sly.logger.info(f"Val set: {len(val_set)} images")
        train_len = len(train_set)
        train_set = remove_empty_items(train_set)
        sly.logger.info(f"Found {train_len - len(train_set)} images without labels in train set.")
        sly.logger.info(f"{len(train_set)} / {train_len} images will be included to train.")
        val_len = len(val_set)
        val_set = remove_empty_items(val_set)
        sly.logger.info(f"Found {val_len - len(val_set)} images without labels in validation set.")
        sly.logger.info(f"{len(val_set)} / {val_len} images will be included to validation.")
        
        verify_train_val_sets(train_set, val_set)
        step_done = True
    except Exception as e:
        train_set = None
        val_set = None
        train_len = None
        val_len = None
        step_done = False
        raise e
    finally:
        api.task.set_field(task_id, "state.splitInProgress", False)
        fields = [
            {"field": "state.splitInProgress", "payload": False},
            {"field": f"data.done2", "payload": step_done},
            {"field": f"state.trainImagesCount", "payload": None if train_set is None else len(train_set)},
            {"field": f"state.valImagesCount", "payload": None if val_set is None else len(val_set)},
            {"field": f"state.fullTrainImagesCount", "payload": train_len},
            {"field": f"state.fullValImagesCount", "payload": val_len},
        ]
        if step_done is True:
            fields.extend([
                {"field": "state.collapsed7", "payload": False},
                {"field": "state.disabled7", "payload": False},
                {"field": "state.activeStep", "payload": 7},
            ])
        g.api.app.set_fields(g.task_id, fields)
    if train_set is not None:
        _save_set_to_json(train_set_path, train_set)
    if val_set is not None:
        _save_set_to_json(val_set_path, val_set)

def _save_set_to_json(save_path, items):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    res = []
    for item in items:
        res.append({
            "dataset_name": item.dataset_name,
            "item_name": item.name
        })
    sly.json.dump_json_file(res, save_path)