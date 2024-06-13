import numpy as np
import os
import torch
from torch.nn import MSELoss
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
import json
from PIL import Image

import argparse
import random
from torch.utils.tensorboard import SummaryWriter
import cv2
import time
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
    parser.add_argument("--epoch", type=int, default=400,
        help="epoch")
    parser.add_argument("--lr", type=float, default=1e-3,
        help="lr")
    parser.add_argument("--torch-deterministic", type=bool, default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=bool, default=False,
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
    def __init__(self, img_dir,annotations_file, train_flag = True, rgb_cnn = False):
        with open(annotations_file) as f:
            self.img_labels = json.load(f)
        self.len = len(self.img_labels)
        self.img_dir = img_dir
        self.train_flag = train_flag
        self.rgb_cnn = rgb_cnn
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
        label = []
        images = []
        for j in range(1 if self.rgb_cnn else 4):
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
            if self.rgb_cnn:
                image = np.transpose(image,(2, 0, 1))
            if not self.rgb_cnn:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            images.append(image)
        label = torch.tensor(label)
        if not self.rgb_cnn:
            image= np.stack(images)
        image = torch.Tensor(image/255.0)

        return image, label


if __name__ == '__main__':
    #record
    args = parse_args()
    best_acc = 9999
    run_name = f"id{args.env_id}_mass{args.mass_centri_cnn}_single{args.single_frame}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    
    asset_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets")
    images_dir = os.path.join(asset_dir, f'{args.env_id}'+'_masks_train')
    labels = os.path.join(images_dir, 'labels.json')
    images_dir_test = os.path.join(asset_dir, f'{args.env_id}'+'_masks_test')
    labels_test = os.path.join(images_dir_test, 'labels.json')

    batch_size = args.batch_size

    train_dataset = CustomImageDataset(images_dir,labels,train_flag=True,rgb_cnn = args.single_frame)
    test_dataset = CustomImageDataset(images_dir_test,labels_test,train_flag=False,rgb_cnn = args.single_frame)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print(labels_test)
    if args.single_frame:
        if args.mass_centri_cnn:
            model = mass_centre_cnn.get_MCCNN_network_frame().to(device)
        else:
            model = OD_frame().to(device)
    else:
        if args.mass_centri_cnn:
            model = mass_centre_cnn.get_MCCNN_network().to(device)
        else:
            model = OD_frames().to(device)
    
    optm = Adam(model.parameters(), lr=args.lr)
    loss_fn = MSELoss()
    all_epoch = args.epoch
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            optm.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label)
            loss.backward()
            optm.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        acc = 0
        with torch.no_grad():

            for idx, (test_x, test_label) in enumerate(test_loader):
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                predict_y = model(test_x.float()).detach()
                acc = acc + loss_fn(predict_y, test_label)
        print("ground_truth1:")
        print(test_label[55])
        print("pred_cors1:")
        print(predict_y[55])
        print(f'epoch:{current_epoch} loss: {acc/len(test_loader)}')
        if args.track:
            writer.add_scalar("charts/loss", acc/len(test_loader), current_epoch)
            wandb.log({'pred_y':predict_y})
        if not os.path.isdir("models"):
            os.mkdir("models")
        # torch.save(model, 'models/pongstack_{:.3f}.pkl'.format(acc/len(test_loader)))
        if best_acc > acc/len(test_loader):
            best_acc = acc/len(test_loader)
            print("new best!")
            if args.single_frame:
                if args.mass_centri_cnn:
                    torch.save(model, 'models/'+f'{args.env_id}'+'_od_mass_single.pkl')
                else:
                    torch.save(model, 'models/'+f'{args.env_id}'+'_od_single.pkl')
            else:
                if args.mass_centri_cnn:
                    torch.save(model, 'models/'+f'{args.env_id}'+'_od_mass.pkl')
                else:
                    torch.save(model, 'models/'+f'{args.env_id}'+'_od.pkl')
    print(f"end best acc:{best_acc}")
    print("Model finished training")