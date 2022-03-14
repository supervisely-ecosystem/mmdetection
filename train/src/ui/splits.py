from binascii import rlecode_hqx
import supervisely as sly
import sly_globals as g
import os
import os.path as osp
import random
import numpy as np
import mmcv
from itertools import groupby
from collections import namedtuple

ItemInfo = namedtuple('ItemInfo', ['dataset_name', 'name', 'img_path', 'ann_path'])
train_set = None
val_set = None

items_to_ignore: dict = {}

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
    # RITM model requires to contain minimum 2 samples in batch
    if train_count < 2:
        train_count = 2
    elif project_info.items_count - train_count < 2:
        train_count = project_info.items_count - 2
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
    state["doubleTaggedImages"] = "train"
    state["ignoredUntaggedImages"] = 0
    state["ignoredDoubleTaggedImages"] = 0
    state["splitInProgress"] = False
    state["trainImagesCount"] = None
    state["valImagesCount"] = None
    data["doneSplits"] = False
    state["collapsedSplits"] = True
    state["disabledSplits"] = True


def restart(data, state):
    state["collapsedSplits"] = False
    state["disabledSplits"] = False
    data["doneSplits"] = False


def refresh_table():
    global items_to_ignore
    ignored_items_count = sum([len(ds_items) for ds_items in items_to_ignore.values()])
    total_items_count = g.project_fs.total_items - ignored_items_count
    train_percent = 80
    train_count = int(total_items_count / 100 * train_percent)
    # RITM model requires to contain minimum 2 samples in batch
    if train_count < 2:
        train_count = 2
    elif g.project_info.items_count - train_count < 2:
        train_count = g.project_info.items_count - 2
    random_split_tab = {
        "count": {
            "total": total_items_count,
            "train": train_count,
            "val": total_items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    fields = [
        {'field': 'state.randomSplit', 'payload': random_split_tab},
        {'field': 'data.totalImagesCount', 'payload': total_items_count},
    ]
    g.api.app.set_fields(g.task_id, fields)



def get_train_val_splits_by_count(train_count, val_count):
    global items_to_ignore
    ignored_count = sum([len(ds_items) for ds_items in items_to_ignore.values()])
    
    if g.project_fs.total_items != train_count + val_count + ignored_count:
        raise ValueError("total_count != train_count + val_count + ignored_count")
    all_items = []
    for dataset in g.project_fs.datasets:
        for item_name in dataset:
            if item_name in items_to_ignore[dataset.name]:
                continue
            all_items.append(ItemInfo(dataset_name=dataset.name,
                                name=item_name,
                                img_path=dataset.get_img_path(item_name),
                                ann_path=dataset.get_ann_path(item_name)))
    random.shuffle(all_items)
    train_items = all_items[:train_count]
    val_items = all_items[train_count:]
    return train_items, val_items


def get_train_val_splits_by_tag(train_tag_name, val_tag_name, untagged="ignore", double_tagged="ignore"):
    global items_to_ignore
    untagged_actions = ["ignore", "train", "val"]
    double_tagged_actions = ["ignore", "train", "val"]
    if untagged not in untagged_actions:
        raise ValueError(f"Unknown untagged action {untagged}. Should be one of {untagged_actions}")
    if double_tagged not in double_tagged_actions:
        raise ValueError(f"Unknown double tagged action {double_tagged}. Should be one of {double_tagged_actions}")

    train_items = []
    val_items = []
    ignored_untagged_cnt = 0
    ignored_double_tagged_cnt = 0
    for dataset in g.project_fs.datasets:
        for item_name in dataset:
            if item_name in items_to_ignore[dataset.name]:
                continue
            img_path, ann_path = dataset.get_item_paths(item_name)
            info = ItemInfo(dataset.name, item_name, img_path, ann_path)

            ann = sly.Annotation.load_json_file(ann_path, g.project_meta)
            if ann.img_tags.get(train_tag_name) is not None and ann.img_tags.get(val_tag_name) is not None:
                # multiple tagged item
                if double_tagged == "ignore":
                    ignored_double_tagged_cnt += 1
                    continue
                elif double_tagged == "train":
                    train_items.append(info)
                elif double_tagged == "val":
                    val_items.append(info)
            elif ann.img_tags.get(train_tag_name) is not None:
                train_items.append(info)
            elif ann.img_tags.get(val_tag_name) is not None:
                val_items.append(info)
            else:
                # untagged item
                if untagged == "ignore":
                    ignored_untagged_cnt += 1
                    continue
                elif untagged == "train":
                    train_items.append(info)
                elif untagged == "val":
                    val_items.append(info)
    return train_items, val_items, ignored_untagged_cnt, ignored_double_tagged_cnt

def get_train_val_splits_by_dataset(train_datasets, val_datasets):
    def _add_items_to_list(datasets_names, items_list):
        global items_to_ignore
        for dataset_name in datasets_names:
            dataset = g.project_fs.datasets.get(dataset_name)
            if dataset is None:
                raise KeyError(f"Dataset '{dataset_name}' not found")
            for item_name in dataset:
                if item_name in items_to_ignore[dataset.name]:
                    continue
                img_path, ann_path = dataset.get_item_paths(item_name)
                info = ItemInfo(dataset.name, item_name, img_path, ann_path)
                items_list.append(info)

    train_items = []
    _add_items_to_list(train_datasets, train_items)
    val_items = []
    _add_items_to_list(val_datasets, val_items)
    return train_items, val_items


def get_train_val_sets(state):
    split_method = state["splitMethod"]
    if split_method == "random":
        train_count = state["randomSplit"]["count"]["train"]
        val_count = state["randomSplit"]["count"]["val"]
        train_set, val_set = get_train_val_splits_by_count(train_count, val_count)
        return train_set, val_set
    elif split_method == "tags":
        train_tag_name = state["trainTagName"]
        val_tag_name = state["valTagName"]
        add_untagged_to = state["untaggedImages"]
        add_double_tagged_to = state["doubleTaggedImages"]
        train_set, val_set, ignored_untagged_cnt, ignored_double_tagged_cnt = get_train_val_splits_by_tag(train_tag_name, val_tag_name,
                                                                     add_untagged_to, add_double_tagged_to)
        return train_set, val_set, ignored_untagged_cnt, ignored_double_tagged_cnt
    elif split_method == "datasets":
        train_datasets = state["trainDatasets"]
        val_datasets = state["valDatasets"]
        train_set, val_set = get_train_val_splits_by_dataset(train_datasets, val_datasets)
        return train_set, val_set
    else:
        raise ValueError(f"Unknown split method: {split_method}")


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    elif len(train_set) < 2:
        raise ValueError("Train set is not big enough, min size is 1.")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")
    elif len(val_set) < 2:
        raise ValueError("Val set is not big enough, min size is 1.")


@g.my_app.callback("create_splits")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def create_splits(api: sly.Api, task_id, context, state, app_logger):
    step_done = False
    ignored_untagged_cnt = 0
    ignored_double_tagged_cnt = 0
    global train_set, val_set
    try:
        api.task.set_field(task_id, "state.splitInProgress", True)
        if state["splitMethod"] == "tags":
            train_set, val_set, ignored_untagged_cnt, ignored_double_tagged_cnt = get_train_val_sets(state)
        else:
            train_set, val_set = get_train_val_sets(state)
        # TODO: log info about ignored images
        verify_train_val_sets(train_set, val_set)
        step_done = True
    except Exception as e:
        train_set = None
        val_set = None
        step_done = False
        raise e
    finally:
        fields = [
            {"field": "state.splitInProgress", "payload": False},
            {"field": "data.doneSplits", "payload": step_done},
            {"field": "state.trainImagesCount", "payload": None if train_set is None else len(train_set)},
            {"field": "state.valImagesCount", "payload": None if val_set is None else len(val_set)},
            {"field": "state.ignoredUntaggedImages", "payload": ignored_untagged_cnt},
            {"field": "state.ignoredDoubleTaggedImages", "payload": ignored_double_tagged_cnt},
        ]
        if step_done is True:
            fields.extend([
                {"field": "state.collapsedAugs", "payload": False},
                {"field": "state.disabledAugs", "payload": False},
                {"field": "state.activeStep", "payload": 5},
            ])
        g.api.app.set_fields(g.task_id, fields)
    if train_set is not None:
        sly.logger.info("Converting train annotations to COCO format...")
        save_set_to_coco_json(train_set_path, train_set, state["selectedClasses"], state["task"])
    if val_set is not None:
        sly.logger.info("Converting val annotations to COCO format...")
        save_set_to_coco_json(val_set_path, val_set, state["selectedClasses"], state["task"])


def mask_to_image_size(label, existence_mask, img_size):
    mask_in_images_coordinates = np.zeros(img_size, dtype=bool)  # size is (h, w)

    row, column = label.geometry.origin.row, label.geometry.origin.col  # move mask to image space
    mask_in_images_coordinates[row: row + existence_mask.shape[0], column: column + existence_mask.shape[1]] = \
        existence_mask

    return mask_in_images_coordinates


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def save_set_to_coco_json(save_path, items, selected_classes, task):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cats = [{"id": i, "name": k} for i, k in enumerate(selected_classes)]
    cat2label = {k: i for i, k in enumerate(selected_classes)}
    annotations = []
    images = []
    obj_count = 0
    # TODO: add progress to UI
    for idx, item in enumerate(mmcv.track_iter_progress(items)):
        filename = osp.join(item.dataset_name, "img", item.name)
        ann_path = osp.join(g.project_dir, item.dataset_name, "ann", f"{item.name}.json")
        ann = sly.Annotation.load_json_file(ann_path, g.project_meta)
        height, width = ann.img_size[0], ann.img_size[1]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        for label in ann.labels:
            rect: sly.Rectangle = label.geometry.to_bbox()
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=cat2label[label.obj_class.name],
                bbox=[rect.left, rect.top, rect.width, rect.height],
                area=rect.width * rect.height, # TODO: maybe calculate mask area?
                iscrowd=0)
            
            # TODO: check that polygons work in practice
            if isinstance(label.geometry, sly.Polygon) and task == "instance_segmentation":
                label_render = np.zeros(ann.img_size, dtype=np.uint8)
                label.geometry._draw_impl(label_render, 1)
                rle_seg_mask = binary_mask_to_rle(label_render.astype(np.bool))
                data_anno["segmentation"] = rle_seg_mask
            elif isinstance(label.geometry, sly.Bitmap) and task == "instance_segmentation":
                seg_mask = np.asarray(label.geometry.convert(sly.Bitmap)[0].data)
                seg_mask_in_image_coords = mask_to_image_size(label, seg_mask, ann.img_size)
                rle_seg_mask = binary_mask_to_rle(seg_mask_in_image_coords)
                data_anno["segmentation"] = rle_seg_mask
            
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=cats)
    mmcv.dump(coco_format_json, save_path)
