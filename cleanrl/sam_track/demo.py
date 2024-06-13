#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import os
import sys

import cv2
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args,fastsam_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
import json
from tqdm import tqdm
import copy
from FastSAM.fastsam import FastSAM, FastSAMPrompt 

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, default="PongNoFrameskip-v4",
        help="the id of the environment")
    args = parser.parse_args()
    return args


def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))
def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)
def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)

def show_box(masks, ax):
    x0, y0,w, h = masks['bbox'][0], masks['bbox'][1], masks['bbox'][2], masks['bbox'][3]
    # w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
def show_box_from_return(box, ax,i):
    x0, y0,w, h = box[0], box[1], box[2], box[3]
    # w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
    plt.text(x0, y0, str(i))
def plot_cors(pred_mask, size, frame):
    obj_ids = np.unique(pred_mask)
    obj_ids = obj_ids[obj_ids!=0]
    plt.imshow(colorize_mask(pred_mask))
    objs = {}
    
    for id in obj_ids:
        # Find coordinates where the mask is equal to the current object id
        coords = np.where(pred_mask == id)
        y, x = coords
        
        # Calculate the mean color in the frame within the mask of the current object
        mask = pred_mask == id
        mean_rgb_val = np.mean(frame[mask], axis=0)
        
        # Calculate the center of the object as the mean of x and y coordinates
        cor = np.mean(coords, axis=1)
        plt.text(cor[1], cor[0], str(id), color='red')
        
        # Bounding box
        x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor="blue", linewidth=2, fill=False))
        
        # Normalizing coordinates by size of the frame for consistent representation
        objs[str(id)] = {
            "coordinates": [cor[0] / size[0], cor[1] / size[1]], #y from top to down, x from left to right, from 1 to
            "bounding_box": [x_min / size[1], y_min / size[0], x_max / size[1], y_max / size[0]],
            "rgb_value": mean_rgb_val.tolist()
        }
        
    return objs
def connected_check(remote_masks):
        '''
        Separate remote masks
        '''
        # cv2.imshow('Original Merged Mask', (1-masks) * 255)
        #cv2.waitKey(0)
        unique_ids = np.unique(remote_masks)
        kernel_size = 5 # some some other values
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        for object_id in unique_ids:
            if object_id == 0:
                continue

            binary_mask = (remote_masks == object_id).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary_mask)
            if num_labels>2:
                return True
        return False
def compute_iou(mask1, mask2):
    '''
    Compute the Intersection over Union (IoU) between two binary masks.
    '''
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


# ### Set parameters for input and output

# In[2]:

args = parse_args()
video_name = args.video_name
io_args = {
    'input_video': f'./assets/{video_name}.mp4',
    'output_mask_dir': f'./assets/{video_name}_masks', # save pred masks
    'output_video': f'./assets/{video_name}_seg.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
    'output_gif': f'./assets/{video_name}_seg.gif', # mask visualization

    'output_mask_dir_train': f'./assets/{video_name}_masks_train', # save pred masks
    'output_video_train': f'./assets/{video_name}_seg_train.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
    'output_gif_train': f'./assets/{video_name}_seg_train.gif', # mask visualization

    'output_mask_dir_test': f'./assets/{video_name}_masks_test', # save pred masks
    'output_video_test': f'./assets/{video_name}_seg_test.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
    'output_gif_test': f'./assets/{video_name}_seg_test.gif', # mask visualization
}
print(io_args)


# ### Tuning SAM on the First Frame for Good Initialization

# In[3]:


# choose good parameters in sam_args based on the first frame segmentation result
# other arguments can be modified in model_args.py
# note the object number limit is 255 by default, which requires < 10GB GPU memory with amp
sresolution = 1024 
img_resolution = (sresolution,sresolution)
save_resolution = (512,512)
# img_resolution = (210,160)
min_area = img_resolution[0]*img_resolution[1]/2000
max_area = img_resolution[0]*img_resolution[1]/10
sam_args['generator_args'] = {
        'points_per_side': 32,
        'pred_iou_thresh': 0.9,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 2,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': min_area,
    }
segtracker_args = {
    'sam_gap': 10, # the interval to run sam to segment new objects
    'min_area': min_area, # minimal mask area to add a new mask as a new object
    'max_area': max_area, # minimal mask area to add a new mask as a new object
    'max_obj_num': 256, # maximal object number to track in a video
    'min_new_obj_iou': 0, # the area of a new object iou should < 0% 
    'min_new_obj_iou_disconnected': 0.5 # the area of a new object iou for connnected track masks should < 60% 
}
fastsam_args = {
    "model_path":"ckpt/FastSAM.pt",
    "imgsz":sresolution,
    "retina":True,
    "iou":0.9,
    "conf":0.9,
    "device":'cuda',
}
cap = cv2.VideoCapture(io_args['input_video'])
frame_idx = 0
segtracker = SegTracker(segtracker_args,sam_args,aot_args,fastsam_args)
segtracker.restart_tracker()

with torch.cuda.amp.autocast():
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame,img_resolution)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,10))
        plt.axis('off')
        plt.imshow(frame)
        plt.show()
        pred_mask, boxes = segtracker.seg(frame,return_box=True)
        torch.cuda.empty_cache()
        obj_ids = np.unique(pred_mask)
        obj_ids = obj_ids[obj_ids!=0]
        print("processed frame {}, obj_num {}".format(frame_idx,len(obj_ids)),end='\n')
        break
    cap.release()
    init_res = draw_mask(frame,pred_mask,id_countour=False)
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(init_res)
    plt.show()
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(colorize_mask(pred_mask))
    cors = []
    j = 0
    for id in obj_ids:
        #box
        x0, y0,w, h = boxes[str(id)][0], boxes[str(id)][1], boxes[str(id)][2], boxes[str(id)][3]
        plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
        plt.text(x0, y0, str(id))
        
        cor = np.mean(np.where(pred_mask==id),axis = 1)
        cors.append(cor)
        plt.text(cor[1], cor[0], str(id),color='red')
        j = j+1

    plt.savefig('example.png')
    plt.show()

    del segtracker
    torch.cuda.empty_cache()
    gc.collect()


# ### Generate Results for the Whole Video

# In[4]:


# For every sam_gap frames, we use SAM to find new objects and add them for tracking
# larger sam_gap is faster but may not spot new objects in time


# source video to segment
cap = cv2.VideoCapture(io_args['input_video'])
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# output masks
output_dir = io_args['output_mask_dir']
output_dir_train = io_args['output_mask_dir_train']
output_dir_test = io_args['output_mask_dir_test']
if not os.path.exists(output_dir_train):
    os.makedirs(output_dir_train)
if not os.path.exists(output_dir_test):
    os.makedirs(output_dir_test)
pred_list = []
masked_pred_list = []

torch.cuda.empty_cache()
gc.collect()
sam_gap = segtracker_args['sam_gap']
frame_idx = 0
segtracker = SegTracker(segtracker_args,sam_args,aot_args,fastsam_args)
segtracker.restart_tracker()

cors_all_frames = []
cors_all_frames_train = []
cors_all_frames_test = []

pbar = tqdm(total=frame_count)
with torch.cuda.amp.autocast():
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.resize(frame,img_resolution)

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        size = frame.shape
        if frame_idx == 0:
            pred_mask = segtracker.seg(frame)
            torch.cuda.empty_cache()
            gc.collect()

            segtracker.add_reference(frame, pred_mask)
        elif (frame_idx % sam_gap) == 0:
            seg_mask = segtracker.seg(frame)
            torch.cuda.empty_cache()
            gc.collect()
            track_mask = segtracker.track(frame)
            #sometimes we will not find objs
            if seg_mask is None:
                seg_mask = track_mask
            # find new objects, and update tracker with new objects
            new_obj_mask = segtracker.find_new_objs(track_mask,seg_mask)
            overlap = (track_mask>0) & (new_obj_mask>0)
            # In overlapping areas, we retain the ID of track_mask; in non-overlapping areas, if there are new objects in new_obj_mask, we use the ID of new_obj_mask.
            pred_mask = np.where(overlap, new_obj_mask, track_mask + new_obj_mask * (new_obj_mask > 0))

            if frame_idx < int(frame_count*0.8):
                plot_cors(new_obj_mask,size,frame)
                plt.savefig(io_args['output_mask_dir_train']+'/'+str(frame_idx)+'_new.png')
                plt.clf()
                plot_cors(seg_mask,size,frame)
                plt.savefig(io_args['output_mask_dir_train']+'/'+str(frame_idx)+'_seg.png')
            else:
                plot_cors(new_obj_mask,size,frame)
                plt.savefig(io_args['output_mask_dir_test']+'/'+str(frame_idx-int(frame_count*0.8))+'_new.png')
                plt.clf()
                plot_cors(seg_mask,size,frame)
                plt.savefig(io_args['output_mask_dir_test']+'/'+str(frame_idx-int(frame_count*0.8))+'_seg.png')
            plt.clf()

            segtracker.add_reference(frame, pred_mask)
        else:
            track_mask = segtracker.track(frame,update_memory=True)
            pred_mask = track_mask
        torch.cuda.empty_cache()
        gc.collect()
        #my save plot corrinate
        cors = plot_cors(pred_mask,size,frame)
        cors_all_frames.append(cors)
        if frame_idx < int(frame_count*0.8):
            cors_all_frames_train.append(cors)
            with open(io_args['output_mask_dir_train']+'/'+"labels.json",'w',encoding='utf-8') as json_file:
                json.dump(cors_all_frames_train,json_file,ensure_ascii=False)
        else:
            cors_all_frames_test.append(cors)
            with open(io_args['output_mask_dir_test']+'/'+"labels.json",'w',encoding='utf-8') as json_file:
                json.dump(cors_all_frames_test,json_file,ensure_ascii=False)


        if frame_idx < int(frame_count*0.8):
            plt.savefig(io_args['output_mask_dir_train']+'/'+str(frame_idx)+'.png')
        else:
            plt.savefig(io_args['output_mask_dir_test']+'/'+str(frame_idx-int(frame_count*0.8))+'.png')
        plt.clf()
        
        frame = cv2.resize(frame,save_resolution)
        if frame_idx < int(frame_count*0.8):
            cv2.imwrite(io_args['output_mask_dir_train']+'/'+str('frame')+str(frame_idx)+'.png', cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(io_args['output_mask_dir_test']+'/'+str('frame')+str(frame_idx-int(frame_count*0.8))+'.png', cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        frame = cv2.resize(frame,img_resolution)
        pred_list.append(pred_mask)
        
        # Get the current total number of objects
        current_obj_num = segtracker.get_obj_num()
        # Update progress bar description
        pbar.set_description(f"Processed frame {frame_idx}, obj_num {current_obj_num}")
        pbar.update(1)
        frame_idx += 1

    cap.release()
    print('\nfinished')


# Generate dataset only use track model

# In[ ]:


pbar.close()
# source video to segment
cap = cv2.VideoCapture(io_args['input_video'])
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

pred_list = []
masked_pred_list = []

torch.cuda.empty_cache()
gc.collect()
frame_idx = 0

cors_all_frames = []
cors_all_frames_train = []
cors_all_frames_test = []

pbar = tqdm(total=frame_count)
with torch.cuda.amp.autocast():
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.resize(frame,img_resolution)

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        size = frame.shape
        pred_mask = segtracker.track(frame,update_memory=True)
        torch.cuda.empty_cache()
        gc.collect()

        cors = plot_cors(pred_mask,size,frame)
        cors_all_frames.append(cors)
        if frame_idx < int(frame_count*0.8):
            cors_all_frames_train.append(cors)
            with open(io_args['output_mask_dir_train']+'/'+"labels.json",'w',encoding='utf-8') as json_file:
                json.dump(cors_all_frames_train,json_file,ensure_ascii=False)
        else:
            cors_all_frames_test.append(cors)
            with open(io_args['output_mask_dir_test']+'/'+"labels.json",'w',encoding='utf-8') as json_file:
                json.dump(cors_all_frames_test,json_file,ensure_ascii=False)


        if frame_idx < int(frame_count*0.8):
            plt.savefig(io_args['output_mask_dir_train']+'/'+str(frame_idx)+'.png')
        else:
            plt.savefig(io_args['output_mask_dir_test']+'/'+str(frame_idx-int(frame_count*0.8))+'.png')
        plt.clf()

        frame = cv2.resize(frame,save_resolution)
        if frame_idx < int(frame_count*0.8):
            cv2.imwrite(io_args['output_mask_dir_train']+'/'+str('frame')+str(frame_idx)+'.png', cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(io_args['output_mask_dir_test']+'/'+str('frame')+str(frame_idx-int(frame_count*0.8))+'.png', cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        frame = cv2.resize(frame,img_resolution)
        pred_list.append(pred_mask)
        
        # Get the current total number of objects
        current_obj_num = segtracker.get_obj_num()
        # Update progress bar description
        pbar.set_description(f"Processed frame {frame_idx}, obj_num {current_obj_num}")
        pbar.update(1)
        frame_idx += 1

    cap.release()
    print('\nfinished')


# ### Save results for visualization

# In[ ]:


import cv2
import os

def images_to_video(folder_path, output_path, fps):
     # Get all the image file names in the folder and sort them numerically
     filenames = [os.path.join(folder_path, f"{i}.png") for i in range(1000)]
     # filenames.sort()

     # Read the first image to get the width and height of the video
     frame = cv2.imread(filenames[0])
     h, w, layers = frame.shape
     size = (w, h)

     # Create a video writing object using the XVID codec and MP4V container
     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

     # Add image to video
     for filename in filenames:
         img = cv2.imread(filename)
         out.write(img)

     out.release()

gname = video_name
folder_name = gname+'_masks_train'
folder_path = 'assets/'+folder_name # Your folder path, assuming it is the current directory
output_path = 'assets/'+gname+'_seg.mp4' # Output video file name
fps = 20 # The video frame rate you want

images_to_video(folder_path, output_path, fps)


# In[ ]:


# # save colorized masks as a gif
# imageio.mimsave(io_args['output_gif'],pred_list,duration=duration)
# print("{} saved".format(io_args['output_gif']))


# In[ ]:


# manually release memory (after cuda out of memory)
del segtracker
torch.cuda.empty_cache()
gc.collect()

