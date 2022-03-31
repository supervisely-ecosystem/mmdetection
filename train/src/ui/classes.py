import splits
import supervisely_lib as sly
import sly_globals as g

selected_classes = None

def init(api: sly.Api, data, state, project_id, project_meta: sly.ProjectMeta):
    stats = api.project.get_stats(project_id)
    class_images = {}
    for item in stats["images"]["objectClasses"]:
        class_images[item["objectClass"]["name"]] = item["total"]

    class_objects = {}
    for item in stats["objects"]["items"]:
        class_objects[item["objectClass"]["name"]] = item["total"]

    class_area = {}
    for item in stats["objectsArea"]["items"]:
        class_area[item["objectClass"]["name"]] = round(item["total"], 2)

    selected_classes_json = []
    for obj_class in project_meta.obj_classes:
        obj_class: sly.ObjClass
        if obj_class.geometry_type not in [sly.Point, sly.PointLocation]:
            selected_classes_json.append(obj_class.to_json())

    for obj_class in selected_classes_json:
        obj_class["imagesCount"] = class_images[obj_class["title"]]
        obj_class["objectsCount"] = class_objects[obj_class["title"]]
        obj_class["areaPercent"] = class_area[obj_class["title"]]

    unlabeled_count = 0
    for ds_counter in stats["images"]["datasets"]:
        unlabeled_count += ds_counter["imagesNotMarked"]

    data["classes"] = selected_classes_json
    state["selectedClasses"] = []
    state["classes"] = len(selected_classes_json) * [True]
    data["unlabeledCount"] = unlabeled_count
    state["ignoredItems"] = 0
    state["totalItems"] = g.project_info.items_count
    state["findingItemsToIgnore"] = False
    data["doneClasses"] = False
    state["collapsedClasses"] = True
    state["disabledClasses"] = True


def reload_classes(task):
    stats = g.api.project.get_stats(g.project_id)

    class_images = {}
    for item in stats["images"]["objectClasses"]:
        class_images[item["objectClass"]["name"]] = item["total"]

    class_objects = {}
    for item in stats["objects"]["items"]:
        class_objects[item["objectClass"]["name"]] = item["total"]

    class_area = {}
    for item in stats["objectsArea"]["items"]:
        class_area[item["objectClass"]["name"]] = round(item["total"], 2)
    
    selected_classes_json = []
    for obj_class in g.project_meta.obj_classes:
        obj_class: sly.ObjClass
        if task == "instance_segmentation":
            if obj_class.geometry_type in [sly.Polygon, sly.Bitmap]:
                selected_classes_json.append(obj_class.to_json())
        elif task == "detection":
            if obj_class.geometry_type not in [sly.Point, sly.PointLocation]:
                selected_classes_json.append(obj_class.to_json())

    for obj_class in selected_classes_json:
        obj_class["imagesCount"] = class_images[obj_class["title"]]
        obj_class["objectsCount"] = class_objects[obj_class["title"]]
        obj_class["areaPercent"] = class_area[obj_class["title"]]

    classes_data = selected_classes_json
    classes_to_select = len(selected_classes_json) * [True]

    fields = [
        {'field': 'data.classes', 'payload': classes_data},
        {'field': 'state.classes', 'payload': classes_to_select}
    ]
    g.api.app.set_fields(g.task_id, fields)


def get_items_to_ignore(selected_classes, task):
    items_to_ignore = {}
    for dataset in g.api.dataset.get_list(g.project_id):
        items_to_ignore[dataset.name] = []
        for ann_info in g.api.annotation.get_list(dataset.id):
            ann_objects = ann_info.annotation["objects"]
            labels_to_include = [label for label in ann_objects if label["classTitle"] in selected_classes]
            if task == "instance_segmentation":
                labels_to_include = [label for label in ann_objects if label["geometryType"] in ["bitmap", "polygon"]]
            if len(labels_to_include) == 0:
                items_to_ignore[dataset.name].append(ann_info.image_name)
    return items_to_ignore


@g.my_app.callback("use_classes")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_classes(api: sly.Api, task_id, context, state, app_logger):
    global selected_classes
    selected_classes = state["selectedClasses"]

    sly.logger.info(f"Project data: {g.project_fs.total_items} images")
    g.api.app.set_field(g.task_id, "state.findingItemsToIgnore", True)
    
    splits.items_to_ignore = get_items_to_ignore(state["selectedClasses"], state["task"])
    ignored_items_count = sum([len(ds_items) for ds_items in splits.items_to_ignore.values()])
    
    sly.logger.info(f"{ignored_items_count} images without selected labels ignored.")
    sly.logger.info(f"{g.project_fs.total_items - ignored_items_count} / {g.project_fs.total_items} images will be included to training.")

    splits.refresh_table()

    fields = [
        {"field": "state.selectedClasses", "payload": state["selectedClasses"]},
        {"field": "state.ignoredItems", "payload": ignored_items_count},
        {"field": "state.findingItemsToIgnore", "payload": False},
        {"field": "data.doneClasses", "payload": True},
        {"field": "state.collapsedSplits", "payload": False},
        {"field": "state.disabledSplits", "payload": False},
        {"field": "state.activeStep", "payload": 5},
    ]
    g.api.app.set_fields(g.task_id, fields)


def restart(data, state):
    data["doneClasses"] = False