import supervisely as sly
from mmdet.datasets.builder import PIPELINES
from mmdet.core.mask.structures import BitmapMasks
import numpy as np


@PIPELINES.register_module()
class SlyImgAugs(object):
    def __init__(self, config_path):
        self.config_path = config_path
        if self.config_path is not None:
            config = sly.json.load_json_file(self.config_path)
            self.augs = sly.imgaug_utils.build_pipeline(config["pipeline"], random_order=config["random_order"])

    
    def _apply_augs(self, results):
        if self.config_path is None:
            return

        img = results["img"]
        boxes = results["gt_bboxes"].tolist()
        if len(results["gt_bboxes"]) == 0:
            return
        float32 = False
        if img.dtype == np.float32:
            float32 = True
        img = img.astype(np.uint8)

        if "gt_masks" in results.keys():
            masks = results["gt_masks"]
            assert isinstance(masks, BitmapMasks)
            masks = masks.masks.transpose((1, 2, 0))

            if "gt_semantic_seg" in results.keys():
                semantic = results["gt_semantic_seg"]
                res_img, res_boxes, res_semantic, res_masks = sly.imgaug_utils.apply_to_image_bbox_and_both_types_masks(self.augs, img, boxes, semantic, masks)
                results["gt_semantic_seg"] = res_semantic
            else:
                res_img, res_boxes, res_masks = sly.imgaug_utils.apply_to_image_bbox_and_mask(self.augs, img, boxes, masks, segmentation_type='instance')
            
            res_masks = res_masks.transpose((2, 0, 1))
            res_masks = BitmapMasks(res_masks, res_masks.shape[1], res_masks.shape[2])
            results["gt_masks"] = res_masks
        else:
            res_img, res_boxes = sly.imgaug_utils.apply_to_image_and_bbox(self.augs, img, boxes)
        if float32:
            res_img = res_img.astype(np.float32)
        res_boxes = np.array(res_boxes, dtype=np.float32)

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