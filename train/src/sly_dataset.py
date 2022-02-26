from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
import supervisely as sly
import sly_globals as g
import numpy as np
import os
import json


@DATASETS.register_module()
class SuperviselyDataset(CustomDataset):
    CLASSES = None
    PALETTE = None

    def __init__(self,
                ann_file,
                pipeline,
                classes=None,
                data_root=None,
                img_prefix=None,
                seg_prefix=None,
                proposal_file=None,
                test_mode=False):
    
        self.data_root = data_root
        self.test_mode = test_mode
        self.CLASSES = self.get_classes(classes)
        self.dataset_samples = self.get_items_by_set_path(ann_file)
        self.data_infos = self.load_annotations()
        self.img_prefix = None
        self.seg_prefix = None
        self.proposal_file = False
        self.proposals = None

        if not test_mode:
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)
    

    def __len__(self):
        return sum([len(items) for dataset, items in self.dataset_samples.items()])


    def load_annotations(self):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}

        data_infos = []
        for ds_name, samples in self.dataset_samples.items():
            for item_name in samples:
                filename = os.path.join(self.data_root, ds_name, "img", item_name)
                ann_path = os.path.join(self.data_root, ds_name, "ann", f'{item_name}.json')

                ann = sly.Annotation.load_json_file(ann_path, g.project_det_meta)
                data_info = dict(filename=filename, width=ann.img_size[1], height=ann.img_size[0])
                
                gt_bboxes = []
                gt_labels = []

                for label in ann.labels:
                    rect: sly.Rectangle = label.geometry.to_bbox()
                    bbox = [rect.left, rect.top, rect.right, rect.bottom]
                    gt_bboxes.append(bbox)
                    class_idx = cat2label[label.obj_class.name]
                    gt_labels.append(class_idx)
                
                data_anno = dict(
                    bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(gt_labels, dtype=np.long)
                )

                data_info.update(ann=data_anno)
                data_infos.append(data_info)

        return data_infos

    
    def get_items_by_set_path(self, set_path):
        files_by_datasets = {}
        with open(set_path, 'r') as set_file:
            set_list = json.load(set_file)

            for row in set_list:
                existing_items = files_by_datasets.get(row['dataset_name'], [])
                existing_items.append(row['item_name'])
                files_by_datasets[row['dataset_name']] = existing_items

        return files_by_datasets
        