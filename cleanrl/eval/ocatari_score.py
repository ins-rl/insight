import pandas as pd
#!/usr/bin/env python
# coding: utf-8
import re
import os
import json

import numpy as np
import pandas as pd
from ocatari.utils import parser

def center_divergence(box1, box2):
    """
    Calculate the center divergence of two bounding boxes.
    """
    # Calculate the center points of two boxes
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)

    # Calculate the Euclidean distance between the center points
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    # Calculate the length of the diagonal of the actual bounding box
    diagonal_length = np.sqrt((box2[2] - box2[0]) ** 2 + (box2[3] - box2[1]) ** 2)

    # Calculate and return the center divergence
    center_divergence = distance / diagonal_length if diagonal_length else float('inf')
    
    return center_divergence

def calculate_tp_fp_fn_no_type(bbox_info_list, img_labels, f_threshold=0.5):
    
    TP = FP = FN = 0

    for detected_bboxes, true_bboxes in zip(bbox_info_list, img_labels):
        matched_true_bboxes = [False] * len(true_bboxes)  
        for detected_bbox in detected_bboxes:
            best_matched_bbox = -1
            best_e_score = 9999
            for i, true_bbox in enumerate(true_bboxes):
                e_score = center_divergence(detected_bbox, true_bbox)
                if e_score <= f_threshold:
                    if e_score<best_e_score:
                        best_matched_bbox = i
                        best_e_score = e_score
            if best_matched_bbox>=0:
                TP += 1
                matched_true_bboxes[best_matched_bbox] = True
            else:
                FP += 1
        FN += matched_true_bboxes.count(False)

    return TP, FP, FN

parser.add_argument("-g", "--game", type=str, default="SpaceInvaders",
                    help="game to evaluate (e.g. 'Pong')")
parser.add_argument("-i", "--interval", type=int, default=1000,
                    help="The frame interval (default 10)")
# parser.add_argument("-m", "--mode", choices=["vision", "ram"],
#                     default="ram", help="The frame interval")
parser.add_argument("-hud", "--hud", action="store_true", default=True, help="Detect HUD")
parser.add_argument("-dqn", "--dqn", action="store_true", default=True, help="Use DQN agent")
opts = parser.parse_args()
prefix = f"{opts.game}_dqn"
file_path = f"path/to/{prefix}.csv"
df = pd.read_csv(f"path/to/{prefix}.csv")
game_name = opts.game+'NoFrameskip-v4'
vis_data = df['VIS']

# Define regular expression matching pattern to extract object type, coordinates and size
pattern_with_size = re.compile(r"(\w+)\s+at\s+\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)")

bbox_info_list = []

for index, vis_item in enumerate(vis_data):
    # Use regular expression to extract object information of current entry
    objects_info = pattern_with_size.findall(vis_item)
    
    # Initialize a dictionary to store bbox information of current entry
    current_entry_info = {"INDEX": index, "BBOXES": []}
    current_entry_info_bbox = []
    
    # Calculate bbox of each object and add to dictionary of current entry
    for obj_info in objects_info:
        obj_type, x, y, w, h = obj_info
        x, y, w, h = map(int, [x, y, w, h])
        bbox = (x, y, (x + w), (y + h)) 
        current_entry_info["BBOXES"].append((obj_type, bbox))
        current_entry_info_bbox.append(bbox)
    
    # Add the current entry's information to the list
    bbox_info_list.append(current_entry_info_bbox)

bbox_df = pd.DataFrame(bbox_info_list)
asset_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'cleanrl','sam_track', "assets")
images_dir = os.path.join(asset_dir, game_name+'_masks_train')
labels = os.path.join(images_dir, 'labels.json')
labels =  os.path.join(f"cleanrl/sam_track/assets", game_name+'_masks_train', 'labels.json')
with open(labels) as f:
    img_labels = json.load(f)

bbox_info_model = []
for i,img_label in enumerate(img_labels):
    current_entry_info_bbox = []
    
    for obj_info in img_label:
        bbox = img_label[obj_info]['bounding_box']
        current_entry_info_bbox.append([bbox[0]*160,bbox[1]*210,bbox[2]*160,bbox[3]*210])
    
    bbox_info_model.append(current_entry_info_bbox)

TP, FP, FN = calculate_tp_fp_fn_no_type(bbox_info_model, bbox_info_list[:8000])

precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0

# 计算F-score
F_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print(f'{opts.game} precision{precision}, recall{recall}, F_score{F_score}')


