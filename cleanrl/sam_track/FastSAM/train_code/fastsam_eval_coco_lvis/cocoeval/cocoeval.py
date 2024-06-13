from pycocotools.coco import COCO
# Added for cross-category evaluation
from cocoeval_wrappers import COCOEvalWrapper, COCOEvalXclassWrapper
import numpy as np
from collections import OrderedDict
import json

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
VOC_CLASSES = (
               'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 
               'motorcycle', 'person', 'potted plant', 'sheep', 'couch',
               'train', 'tv')
NONVOC_CLASSES = (
               'truck', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench',
               'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake',
               'bed', 'toilet', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
class_names_dict = {
        'all': CLASSES,
        'voc': VOC_CLASSES,
        'nonvoc': NONVOC_CLASSES
    }
try:
    import pycocotools
    if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
        assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')
coco = COCO(f'/home/{USER}/detectron2/datasets/coco/annotations/instances_val2017.json')
cocoGt = coco
eval_results = OrderedDict()
f = open(f'/home/{USER}/yolo_sa/sam_h_20_coco_17val.json', 'r')
data = json.load(f)
f.close()
cocoDt = cocoGt.loadRes(data)
eval_cat_ids = coco.get_cat_ids(cat_names=class_names_dict['all'])
for idx, ann in enumerate(cocoGt.dataset['annotations']):
    if ann['category_id'] in eval_cat_ids:
        cocoGt.dataset['annotations'][idx]['ignored_split'] = 0
    else:
        cocoGt.dataset['annotations'][idx]['ignored_split'] = 1
iou_type = 'bbox'
cocoEval = COCOEvalXclassWrapper(cocoGt, cocoDt, iou_type)
cat_ids = coco.get_cat_ids(cat_names=CLASSES)
img_ids = coco.get_img_ids()
cocoEval.params.catIds = cat_ids
cocoEval.params.imgIds = img_ids
proposal_nums=(10, 20, 30, 50, 100, 300, 500, 1000, 1500)
cocoEval.params.maxDets = list(proposal_nums)
iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
cocoEval.params.iouThrs = iou_thrs
coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@10': 6,
                'AR@20': 7,
                'AR@50': 8,
                'AR@100': 9,
                'AR@300': 10,
                'AR@500': 11,
                'AR@1000': 12,
                'AR@1500': 13,
            }
cocoEval.params.useCats = 0  # treat all FG classes as single class.
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
metric_items = ['AR@10', 'AR@20', 'AR@50', 'AR@100', 'AR@300', 'AR@500', 'AR@1000', 'AR@1500']
for metric_item in metric_items:
    key = f'{metric}_{metric_item}'
    val = float(
        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
    eval_results[key] = val
ap = cocoEval.stats[:6]
eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')
print(eval_results)