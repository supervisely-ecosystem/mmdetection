import supervisely as sly
from mmdet.datasets.builder import PIPELINES
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from mmdet.core.mask.structures import BitmapMasks, PolygonMasks
import numpy as np
import datetime
import cv2


@PIPELINES.register_module()
class SlyImgAugs(object):
    def __init__(self, config_path):
        self.config_path = config_path
        if self.config_path is not None:
            config = sly.json.load_json_file(self.config_path)
            self.augs = sly.imgaug_utils.build_pipeline(config["pipeline"], random_order=config["random_order"])
    
    def apply_to_image_and_bbox(self, augs, img, bbox):
        boxes = [BoundingBox(box[0], box[1], box[0] + box[2], box[1] + box[3]) for box in bbox]
        boxes = BoundingBoxesOnImage(boxes, shape=img.shape[:2])
        res_img, res_boxes, _ = sly.imgaug_utils._apply(augs, img, boxes=boxes)
        res_boxes = np.array([[res_box.x1, res_box.y1, res_box.x2 - res_box.x1, res_box.y2 - res_box.y1] for res_box in res_boxes], dtype=np.float32)

        return res_img, res_boxes

    def to_nonoverlapping_masks(self, bitmap_masks):
        common_img = np.zeros(bitmap_masks.shape[1:], np.int32)  # size is (h, w)

        for idx, mask in enumerate(bitmap_masks):
            common_img[mask.astype(np.bool)] = idx
        
        return common_img

    def to_bitmap_masks(self, common_mask, N):
        bitmap_masks = []
        for idx in range(N):
            instance_mask = common_mask == idx
            bitmap_masks.append(instance_mask.astype(np.uint8))
        return np.stack(bitmap_masks)
        
    def apply_to_image_bbox_and_mask(self, augs, img, bbox, masks):
        boxes = [BoundingBox(box[0], box[1], box[0] + box[2], box[1] + box[3]) for box in bbox]
        boxes = BoundingBoxesOnImage(boxes, shape=img.shape[:2])
        if isinstance(masks, BitmapMasks):
            np_masks = self.to_nonoverlapping_masks(masks.masks)
        else:
            raise NotImplementedError()

        segmaps = SegmentationMapsOnImage(np_masks, shape=img.shape[:2])
        res_img, res_boxes, res_segmaps = sly.imgaug_utils._apply(augs, img, boxes=boxes, masks=segmaps)
        np_masks = res_segmaps.get_arr()

        if isinstance(masks, BitmapMasks):
            np_masks = self.to_bitmap_masks(np_masks, len(masks.masks))
        res_masks = BitmapMasks(np_masks, np_masks.shape[1], np_masks.shape[2])
        res_boxes = np.array([[res_box.x1, res_box.y1, res_box.x2 - res_box.x1, res_box.y2 - res_box.y1] for res_box in res_boxes], dtype=np.float32)

        if res_img.shape[:2] != res_masks.masks.shape[-2:]:
            raise ValueError(f"Image and mask have different shapes "
                            f"({res_img.shape[:2]} != {res_masks.masks.shape[-2:]}) after augmentations. "
                            f"Please, contact tech support")
        return res_img, res_boxes, res_masks
    
    def _apply_augs(self, results):
        if self.config_path is None:
            return

        img = results["img"]
        boxes = results["gt_bboxes"]

        if "gt_masks" in results.keys():
            masks = results["gt_masks"]
            res_img, res_boxes, res_masks = self.apply_to_image_bbox_and_mask(self.augs, img, boxes, masks)
            results["gt_masks"] = res_masks
        else:
            res_img, res_boxes = self.apply_to_image_and_bbox(self.augs, img, boxes)
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