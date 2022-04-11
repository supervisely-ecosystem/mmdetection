import supervisely as sly
from mmdet.datasets.builder import PIPELINES
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from mmdet.core.mask.structures import BitmapMasks
from supervisely.sly_logger import logger
import numpy as np
import imgaug.augmenters as iaa


def get_function(category_name, aug_name):
    try:
        submodule = getattr(iaa, category_name)
        aug_f = getattr(submodule, aug_name)
        return aug_f
    except Exception as e:
        logger.error(repr(e))
        # raise e
        return None

def build_pipeline(aug_infos, random_order=False):
    pipeline = []
    for aug_info in aug_infos:
        category_name = aug_info["category"]
        aug_name = aug_info["name"]
        params = aug_info["params"]
        for param_name, param_val in params.items():
            if isinstance(param_val, dict):
                if "x" in param_val.keys() and "y" in param_val.keys():
                    param_val["x"] = tuple(param_val["x"])
                    param_val["y"] = tuple(param_val["y"])
            elif isinstance(param_val, list):
                params[param_name] = tuple(param_val)

        aug_func = get_function(category_name, aug_name)

        aug = aug_func(**params)

        sometimes = aug_info.get("sometimes", None)
        if sometimes is not None:
            aug = iaa.meta.Sometimes(sometimes, aug)
        pipeline.append(aug)
    augs = iaa.Sequential(pipeline, random_order=random_order)
    return augs


def aug_to_python(aug_info):
    pstr = ""
    for name, value in aug_info["params"].items():
        v = value
        if type(v) is list:  #name != 'nb_iterations' and
            v = (v[0], v[1])
        elif type(v) is dict and "x" in v.keys() and "y" in v.keys():
            v = {"x": (v["x"][0], v["x"][1]), "y": (v["y"][0], v["y"][1])}

        if type(value) is str:
            pstr += f"{name}='{v}', "
        else:
            pstr += f"{name}={v}, "
    method_py = f"iaa.{aug_info['category']}.{aug_info['name']}({pstr[:-2]})"

    res = method_py
    if "sometimes" in aug_info:
        res = f"iaa.Sometimes({aug_info['sometimes']}, {method_py})"
    return res


def pipeline_to_python(aug_infos, random_order=False):
    template = \
"""import imgaug.augmenters as iaa

seq = iaa.Sequential([
{}
], random_order={})
"""
    py_lines = []
    for info in aug_infos:
        line = aug_to_python(info)
        _validate = info["python"]
        if line != _validate:
            raise ValueError("Generated python line differs from the one from config: \n\n{!r}\n\n{!r}"
                             .format(line, _validate))
        py_lines.append(line)
    res = template.format('\t' + ',\n\t'.join(py_lines), random_order)
    return res


@PIPELINES.register_module()
class SlyImgAugs(object):
    def __init__(self, config_path):
        self.config_path = config_path
        if self.config_path is not None:
            config = sly.json.load_json_file(self.config_path)
            self.augs = build_pipeline(config["pipeline"], random_order=config["random_order"])

    def apply_to_image_and_bbox(self, augs, img, bbox):
        boxes = [BoundingBox(box[0], box[1], box[2], box[3]) for box in bbox]
        boxes = BoundingBoxesOnImage(boxes, shape=img.shape[:2])
        res_img, res_boxes, _ = sly.imgaug_utils._apply(augs, img, boxes=boxes)
        res_boxes = np.array([[res_box.x1, res_box.y1, res_box.x2, res_box.y2] for res_box in res_boxes], dtype=np.float32)

        return res_img, res_boxes

    def to_nonoverlapping_masks(self, bitmap_masks):
        common_img = np.zeros(bitmap_masks.shape[1:], np.int32)  # size is (h, w)

        for idx, mask in enumerate(bitmap_masks):
            common_img[mask.astype(np.bool)] = idx + 1
        
        return common_img

    def to_bitmap_masks(self, common_mask, N):
        bitmap_masks = []
        for idx in range(N):
            instance_mask = common_mask == idx + 1
            bitmap_masks.append(instance_mask.astype(np.uint8))
        return np.stack(bitmap_masks)
        
    def apply_to_image_bbox_and_mask(self, augs, img, bbox, masks, semantic=None):
        # bbox format: [x1, y1, x2, y2]
        boxes = [BoundingBox(box[0], box[1], box[2], box[3]) for box in bbox]
        boxes = BoundingBoxesOnImage(boxes, shape=img.shape[:2])
        if isinstance(masks, BitmapMasks):
            np_masks = self.to_nonoverlapping_masks(masks.masks)
            if semantic is not None:
                # 3-dim image
                np_masks = np.stack((np_masks, semantic, semantic), axis=-1)
            else:
                np_masks = np_masks[:,:,np.newaxis]
        else:
            raise NotImplementedError()

        segmaps = SegmentationMapsOnImage(np_masks, shape=np_masks.shape)
        res_img, res_boxes, res_segmaps = sly.imgaug_utils._apply(augs, img, boxes=boxes, masks=segmaps)
        np_masks = res_segmaps.get_arr()

        if semantic is not None:
            res_semantic = np_masks[:,:,2]
        if isinstance(masks, BitmapMasks):
            np_masks = self.to_bitmap_masks(np_masks[:,:,0], len(masks.masks))
        res_masks = BitmapMasks(np_masks, np_masks.shape[1], np_masks.shape[2])
        res_boxes = np.array([[res_box.x1, res_box.y1, res_box.x2, res_box.y2] for res_box in res_boxes], dtype=np.float32)

        if res_img.shape[:2] != res_masks.masks.shape[-2:]:
            raise ValueError(f"Image and mask have different shapes "
                            f"({res_img.shape[:2]} != {res_masks.masks.shape[-2:]}) after augmentations. "
                            f"Please, contact tech support")
        if semantic is not None:
            return res_img, res_boxes, res_masks, res_semantic
        return res_img, res_boxes, res_masks
    
    def _apply_augs(self, results):
        if self.config_path is None:
            return

        img = results["img"]
        boxes = results["gt_bboxes"]
        if len(results["gt_bboxes"]) == 0:
            return
        float32 = False
        if img.dtype == np.float32:
            float32 = True
        img = img.astype(np.uint8)

        if "gt_masks" in results.keys():
            masks = results["gt_masks"]
            if "gt_semantic_seg" in results.keys():
                semantic = results["gt_semantic_seg"]
                res_img, res_boxes, res_masks, res_semantic = self.apply_to_image_bbox_and_mask(self.augs, img, boxes, masks, semantic)
                results["gt_semantic_seg"] = res_semantic
            else:
                res_img, res_boxes, res_masks = self.apply_to_image_bbox_and_mask(self.augs, img, boxes, masks)
            results["gt_masks"] = res_masks
        else:
            res_img, res_boxes = self.apply_to_image_and_bbox(self.augs, img, boxes)
        if float32:
            res_img = res_img.astype(np.float32)
        
        results["img"] = res_img
        results["gt_bboxes"] = res_boxes
        results['img_shape'] = res_img.shape

    def __call__(self, results):
        self._apply_augs(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(config_path={self.config_path})'
        return repr_str