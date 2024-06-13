import numpy as np
import os

import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
import json
from PIL import Image

import argparse
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import matplotlib.pyplot as plt
import cv2
import torchvision.models as models

from cleanrl.agents.Normal_Cnn import Model,OD_frame, OD_frames
from cleanrl.agents import mass_centre_cnn

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--batch-size", type=int, default=512,
        help="batch-size")
    parser.add_argument("--lr", type=int, default=1e-3,
        help="lr")
    parser.add_argument("--torch-deterministic", type=bool, default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=bool, default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Cnn_location_extra",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--mass_centri_cnn", type=bool, default=False,
        help="use mass_centri_cnn or not")
    parser.add_argument("--single_frame", type=bool, default=False,
        help="single frame or not")
    args = parser.parse_args()
    return args


class CustomImageDataset(Dataset):
    def __init__(self, img_dir,annotations_file,train_flag = True):
        with open(annotations_file) as f:
            self.img_labels = json.load(f)
        self.len = len(self.img_labels)
        self.img_dir = img_dir
        self.train_flag = train_flag
        for frame_dix in range(len(self.img_labels)):
            for i in range(25):
                if str(i) not in self.img_labels[frame_dix]:
                    if frame_dix!=0:
                        self.img_labels[frame_dix][str(i)] = self.img_labels[frame_dix-1][str(i)] 
                    else:
                        self.img_labels[frame_dix][str(i)] = [0,0]


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'frame'+str(idx)+'.png')
        image = Image.open(img_path).convert('RGB')
        plt.figure(figsize=(10,10))
        plt.axis('off')
        plt.imshow(image)
        label = []
        images = []
        for j in range(1 if args.single_frame else 4):
            rn = j
            fin_index = idx+rn
            for objid in range(16):
                label = label + self.img_labels[(fin_index)%self.len][str(objid)]
            img_path = os.path.join(self.img_dir, 'frame'+str((fin_index)%self.len)+'.png')
            image = cv2.imread(img_path)
            #turn from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image, (84,84), interpolation=cv2.INTER_AREA
            )
            if args.single_frame:
                image = np.transpose(image,(2, 0, 1))
            if not args.single_frame:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            images.append(image)
        label = torch.tensor(label)
        if not args.single_frame:
            image= np.stack(images)
        image = torch.Tensor(image/255.0)

        return image, label


if __name__ == '__main__':
    #record
    args = parse_args()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    asset_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets")
    images_dir = os.path.join(asset_dir, f'{args.env_id}'+'_masks_train')
    labels = os.path.join(images_dir, 'labels.json')
    images_dir_test = os.path.join(asset_dir, f'{args.env_id}'+'_masks_test')
    labels_test = os.path.join(images_dir_test, 'labels.json')
    os.makedirs("stack_cnn_out_frames", exist_ok=True)
    batch_size = 1

    test_dataset = CustomImageDataset(images_dir_test,labels_test,train_flag=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if args.single_frame:
        if args.mass_centri_cnn:
            model = torch.load('models/'+f'{args.env_id}'+'_od_mass_single.pkl')
        else:
            model = torch.load( 'models/'+f'{args.env_id}'+'_od_single.pkl')
    else:
        if args.mass_centri_cnn:
            model = torch.load('models/'+f'{args.env_id}'+'_od_mass.pkl')
        else:
            model = torch.load('models/'+f'{args.env_id}'+'_od.pkl')

    optm = Adam(model.parameters(), lr=args.lr)
    loss_fn = MSELoss()
    all_epoch = 1
    for current_epoch in range(all_epoch):
        model.eval()
        acc = 0
        with torch.no_grad():

            for idx, (test_x, test_label) in enumerate(test_loader):
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                predict_y = model(test_x.float()).detach()
                acc = acc + loss_fn(predict_y, test_label)
                test_vis = test_x[0].permute(1,2,0).cpu().numpy()
                # size = test_vis.shape
                size = (210,160)
                # test_vis = test_vis[:,:,0]
                # test_vis = cv2.cvtColor(test_vis, cv2.COLOR_GRAY2RGB)
                # plt.figure(figsize=(10,10))
                # plt.axis('off')
                # plt.imshow(test_vis)
                cors = test_label.cpu().numpy()
                pred_cors = predict_y.cpu().numpy()
                for i in range(16):
                    cors[0][i*2] = cors[0][i*2]*size[0]
                    cors[0][i*2+1] = cors[0][i*2+1]*size[1]
                    pred_cors[0][i*2] = pred_cors[0][i*2]*size[0]
                    pred_cors[0][i*2+1] = pred_cors[0][i*2+1]*size[1]
                    plt.text(cors[0][i*2+1], cors[0][i*2], str(i), fontsize=36,color='red')
                    plt.text(pred_cors[0][i*2+1], pred_cors[0][i*2], str(i), fontsize=36,color='green')
                plt.savefig(os.path.join('stack_cnn_out_frames',f'{idx}.png'))
                plt.close()
        print("ground_truth1:")
        print(test_label[0])
        print("pred_cors1:")
        print(predict_y[0])
        print(f'epoch:{current_epoch} loss: {acc/len(test_loader)}')
    print("Model finished test")