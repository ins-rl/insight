import numpy as np
import os
from torch.utils.data import Dataset
import json
import cv2


class StudentCNNDataset(Dataset):
    def __init__(self, img_dir, max_objects=256, resolution=[64, 64]):
        self.img_dir = img_dir
        with open(os.path.join(img_dir, "labels.json")) as f:
            img_labels = json.load(f)
        self.n_frames = len(img_labels)
        self.coordinates = np.zeros(
            (self.n_frames, max_objects, 2), dtype=np.float32)
        self.presence = np.zeros(
            (self.n_frames, max_objects), dtype=np.float32)
        self.frames = []
        for frame_ind in range(self.n_frames):
            label_dict = img_labels[frame_ind]
            for object_id, coordinates in label_dict.items():
                object_id = int(object_id) - 1
                self.presence[frame_ind, object_id] = 1
                self.coordinates[frame_ind, object_id] = np.array(coordinates)
            img_path = os.path.join(self.img_dir, f'frame{frame_ind}.png')
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image, resolution, interpolation=cv2.INTER_AREA)
            self.frames.append(image)
        self.frames = np.array(self.frames)
    
    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        return self.frames[idx], self.presence[idx], self.coordinates[idx]

