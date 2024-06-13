from agents.Normal_Cnn import OD_frames_gray, OD_frames, OD_frames_gray2
import numpy as np
import os
import torch
from torch.nn import MSELoss
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
import json
from PIL import Image
from sklearn.metrics import average_precision_score
import argparse
import random
from torch.utils.tensorboard import SummaryWriter
import cv2
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualize_utils import images_to_video
from distutils.util import strtobool
from collections import defaultdict
import warnings

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--batch-size", type=int, default=32,
        help="batch-size")
    parser.add_argument("--epoch", type=int, default=600,
        help="epoch")
    parser.add_argument("--lr", type=float, default=3e-4,
        help="lr")
    parser.add_argument("--torch-deterministic", type=bool, default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=bool, default=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Train_CNN",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--mass_centri_cnn", type=bool, default=False,
        help="use mass_centri_cnn or not")
    parser.add_argument("--single_frame", type=bool, default=False,
        help="single frame or not")
    parser.add_argument("--resolution", type=int, default=84,
        help="resolution")
    parser.add_argument("--cors", type=lambda x: bool(strtobool(x)), default=True,
        help="use cors")
    parser.add_argument("--bbox", type=lambda x: bool(strtobool(x)), default=False,
        help="use bbox")
    parser.add_argument("--rgb", type=lambda x: bool(strtobool(x)), default=False,
        help="use rgb")
    parser.add_argument("--obj_vec_length", type=int, default=2,
        help="obj vector length")
    parser.add_argument("--gray", type=lambda x: bool(strtobool(x)), default=True,
        help="use gray or not")
    parser.add_argument("--n_objects", type=int, default=256,
        help="n_objects")
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=False,
        help="resume from existing ckpt.")
    parser.add_argument("--run-name", type=str, default=None,
        help="the defined run_name")
    parser.add_argument("--coordinate_loss", type=str, default="l1")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label_weighting_beta", type=float, default=10)
    args = parser.parse_args()
    args.obj_vec_length = args.cors*2+args.bbox*4+args.rgb*3
    args.cnn_out_dim = args.n_objects*args.obj_vec_length*4
    return args

def binary_focal_loss_with_logits(logits, label, reduction='none', gamma=2):
    # logits: batch, n_label
    log_prob = F.logsigmoid(logits)
    prob = F.sigmoid(logits) ** gamma
    log_prob_ = F.logsigmoid(-logits)
    prob_ = F.sigmoid(-logits) ** gamma
    loss = - (prob_ * log_prob * label + prob * log_prob_ * (1-label))
    if reduction == 'none':
        return loss
    else:
        raise NotImplementedError

def shift_stack_frames(stack_frames, stack_bbox, stack_coordinates):
    existence = (np.max(stack_coordinates, -1) > 0)
    stack_bbox = stack_bbox[existence]
    x_min = stack_bbox[:, 0].min()
    y_min = stack_bbox[:, 1].min()
    x_max = stack_bbox[:, 2].max()
    y_max = stack_bbox[:, 3].max()
    to_left = x_min
    to_right = 1 - x_max
    to_top = y_min
    to_bottom = 1 - y_max
    x = np.random.uniform(-to_left/2, to_right/2)
    y = np.random.uniform(-to_top/2, to_bottom/2)
    M = np.float32([[1, 0, x * stack_frames.shape[1]], [0, 1, y * stack_frames.shape[0]]])
    shift_img = cv2.warpAffine(stack_frames.copy(), M, (stack_frames.shape[1], stack_frames.shape[0]))
    stack_coordinates = stack_coordinates.copy()
    stack_coordinates[existence] += np.array([[x, y]])
    shift_coordinates = np.clip(stack_coordinates, 0, 1)
    return shift_img, shift_coordinates

def max_and_skip(frames, coordiantes, bboxes, skip=4):
    out_frames = []
    out_coordinates = []
    out_bboxes = []
    obs_buffer = np.zeros((2, *frames[0].shape))
    for i in range(0, len(frames), skip):
        if i+skip-2 > len(frames) - 1:
            break
        obs_buffer[0] = frames[i+skip-2]
        obs_buffer[1] = frames[i+skip-1]
        out_frames.append(obs_buffer.max(0))
        out_coordinates.append(coordiantes[i+skip-1])
        out_bboxes.append(bboxes[i+skip-1])
    return out_frames, out_coordinates, out_bboxes


class CustomImageDataset(Dataset):
    def __init__(self, img_dir,annotations_file,args, train_flag = False, beta=10):
        with open(annotations_file) as f:
            self.img_labels = json.load(f)
        self.img_dir = img_dir
        self.train_flag = train_flag
        self.resolution = args.resolution
        self.gray = args.gray
        
        self.n_objs = args.n_objects
        self.load_data()
        self.compute_object_weight(beta=beta)
        self.stack_data()

    def load_data(self):
        self.max_objs = 0
        coordinates = []
        bboxes = []
        images = []
        for frame_idx in range(len(self.img_labels)):
            frame_coordinates = []
            frame_bboxes = []
            for i in range(1,self.n_objs+1):
                if str(i) not in self.img_labels[frame_idx]:
                    self.img_labels[frame_idx][str(i)] = {
            "coordinates": [0,0],
            "bounding_box": [0,0,0,0],
            "rgb_value": [0,0,0]}
                obj = self.img_labels[frame_idx][str(i)]
                if obj !={
            "coordinates": [0,0],
            "bounding_box": [0,0,0,0],
            "rgb_value": [0,0,0]
        } and self.max_objs<i:
                    self.max_objs = i
                frame_coordinates.append(obj['coordinates'])
                frame_bboxes.append(obj['bounding_box'])
                # [rgb/255 for rgb in obj['rgb_value']] * self.args.rgb
            coordinates.append(frame_coordinates)
            bboxes.append(frame_bboxes)
            img_path = os.path.join(self.img_dir, f'frame{frame_idx}.png')
            image = cv2.imread(img_path)
            #turn from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        #images, coordinates, bboxes = max_and_skip(images, coordinates, bboxes)
        # resize and gray scale
        images_ = []
        for image in images:
            image = cv2.resize(
                image, (self.resolution,self.resolution),
                interpolation=cv2.INTER_AREA)
            if self.gray:
                image = np.sum(np.multiply(image, np.array([0.2125, 0.7154, 0.0721])), axis=-1)
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            images_.append(image)
        images = images_

        self.coordinates = np.array(coordinates)
        self.bboxes = np.array(bboxes)
        self.images = np.array(images)

    def stack_data(self):
        images = np.concatenate([self.images, self.images[:3]], 0)
        self.stack_frames = np.stack(
            [images[:-3], images[1:-2], images[2:-1], images[3:]], 1)
        temp_coordinates = np.concatenate(
            [self.coordinates, self.coordinates[:3]], 0)
        self.stack_coordinates = np.stack(
            [temp_coordinates[:-3], temp_coordinates[1:-2], temp_coordinates[2:-1], temp_coordinates[3:]], 1)
        temp_bbox = np.concatenate(
            [self.bboxes, self.bboxes[:3]], 0)
        self.stack_bboxes = np.stack(
            [temp_bbox[:-3], temp_bbox[1:-2], temp_bbox[2:-1], temp_bbox[3:]], 1)
        temp_label_weight = np.concatenate(
            [self.sample_label_weight, self.sample_label_weight[:3]], 0)
        self.stack_sample_label_weight = np.stack(
            [temp_label_weight[:-3], temp_label_weight[1:-2], temp_label_weight[2:-1], temp_label_weight[3:]], 1)
    
    def compute_object_weight(self, alpha=0.1, beta=10):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # only support coordiantes for now
        labels = self.coordinates.copy()
        existence = (np.max(labels, -1) > 0).astype(np.float32)
        object_counts = existence.sum(0)
        label_weight = np.where(
            object_counts > 0, 1 / (object_counts + 1e-20), 0) / self.n_objs
        mu = label_weight[label_weight>0].mean()
        if beta > 0:
            sample_label_weight = label_weight[None, ] * existence
            label_weight_norm = sample_label_weight.sum(1, keepdims=True)
            sample_label_weight /= (1e-20 + label_weight_norm)
            sample_label_weight = alpha + sigmoid(beta * (sample_label_weight - mu))
        else:
            sample_label_weight = np.ones_like(label_weight[None, ]) * existence
        self.sample_label_weight = sample_label_weight

        
    def __len__(self):
        return len(self.stack_frames)

    def __getitem__(self, idx):
        images = self.stack_frames[idx].copy()
        coordinates = self.stack_coordinates[idx].copy()
        bboxes = self.stack_bboxes[idx].copy()
        height = bboxes[:, :, 3] - bboxes[:, :, 1]
        width = bboxes[:, :, 2] - bboxes[:, :, 0]
        shape = np.concatenate([height, width], -1)
        images = torch.Tensor(images/255.0)
        coordinates = torch.Tensor(coordinates).flatten(0)
        label_weight = torch.Tensor(self.stack_sample_label_weight[idx]).flatten(0)
        shape = torch.Tensor(shape).flatten(0)
        return images, coordinates, label_weight, shape

def coordinate_label_to_existence_label(
        coordinate_labels,
        n_frame=4,
        n_objs=256,
        obj_vec_length=2):
    batch_size = coordinate_labels.shape[0]
    existence_label = coordinate_labels.reshape(
                (batch_size, n_objs * n_frame, -1))
    existence_label = ((torch.max(existence_label, -1)[0] > 0)
                       .to(torch.float))
    existence_mask = (existence_label.unsqueeze(-1)
                      .repeat(1, 1, obj_vec_length)
                      .flatten(start_dim=1))
    return existence_label, existence_mask

if __name__ == '__main__':
    #record
    args = parse_args()
    best_acc = 9999
    best_avg_precision = 0
    if args.run_name is None:
        run_name = f"id:{args.env_id}-resolution:{args.resolution}-{args.obj_vec_length}-seed{args.seed}-gray{args.gray}-objs{args.n_objects}"
    else:
        run_name = args.run_name
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),)

    
    asset_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'cleanrl','sam_track', "assets")
    images_dir = os.path.join(asset_dir, f'{args.env_id}'+'_masks_train')
    labels = os.path.join(images_dir, 'labels.json')
    images_dir_test = os.path.join(asset_dir, f'{args.env_id}'+'_masks_test')
    labels_test = os.path.join(images_dir_test, 'labels.json')

    batch_size = args.batch_size

    train_dataset = CustomImageDataset(images_dir,labels,args,train_flag=True,beta=args.label_weighting_beta)
    test_dataset = CustomImageDataset(images_dir_test,labels_test,args,train_flag=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=8, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=8, pin_memory=True)
    if args.gray:
        model = OD_frames_gray2(args).to(device)
    else:
        model = OD_frames(args).to(device)
    if args.resume:
        model = torch.load('models/'+f'{args.env_id}'+f'{args.resolution}'+f'{args.obj_vec_length}'+f"_gray{args.gray}"+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'_od.pkl')
    
    optm = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.coordinate_loss == "l2":
        coordinate_loss_fn = torch.nn.functional.mse_loss
    elif args.coordinate_loss == "l1":
        coordinate_loss_fn = torch.nn.functional.l1_loss
    else:
        raise NotImplementedError
    
    all_epoch = args.epoch
    start_time = time.time() 
    for current_epoch in range(all_epoch):
        model.train()
        epoch_start_time = time.time()
        all_coordinate_loss = 0
        all_existence_loss = 0
        all_shape_loss = 0
        for idx, (train_x, train_label, train_label_weight, train_shape) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            train_label_weight = train_label_weight.to(device)
            train_shape = train_shape.to(device)
            existence_label, existence_mask = coordinate_label_to_existence_label(train_label)
            train_label_weight_mask = train_label_weight.unsqueeze(-1).repeat(1, 1, args.obj_vec_length).flatten(start_dim=1)
            optm.zero_grad()
            predict_y, existence_logits, predict_shape = model(train_x.float(), return_existence_logits=True, clip_coordinates=False, return_shape=True)
            coordinate_loss = (coordinate_loss_fn(predict_y, train_label, reduction='none') * existence_mask * train_label_weight_mask).sum(1).mean(0)
            shape_loss = (coordinate_loss_fn(predict_shape, train_shape, reduction='none') * existence_mask * train_label_weight_mask).sum(1).mean(0)
            existence_loss = binary_focal_loss_with_logits(existence_logits, existence_label, reduction='none')
            existence_loss = (existence_loss * train_label_weight).sum(1).mean(0)
            loss = coordinate_loss + existence_loss + shape_loss
            loss.backward()
            optm.step()
            all_coordinate_loss += coordinate_loss.detach()
            all_existence_loss += existence_loss.detach()
            all_shape_loss += shape_loss.detach()
        all_coordinate_loss /= len(train_loader)
        all_existence_loss /= len(train_loader)
        all_shape_loss /= len(train_loader)
        if args.track:
            writer.add_scalar("charts/train_coordinate_loss",
                              all_coordinate_loss, current_epoch)
            writer.add_scalar("charts/train_existence_loss",
                              all_existence_loss, current_epoch)
            writer.add_scalar("charts/train_shape_loss",
                              all_shape_loss, current_epoch)       
        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        acc = 0
        existence_labels = []
        existence_probs = []
        with torch.no_grad():
            for idx, (test_x, test_label, _, _) in enumerate(test_loader):
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                existence_label, existence_mask = coordinate_label_to_existence_label(test_label)
                predict_y, existence_logits = model(test_x.float(), return_existence_logits=True, threshold=0.5)
                predict_y = predict_y.detach()
                existence_prob = F.sigmoid(existence_logits)
                acc = acc + (coordinate_loss_fn(predict_y, test_label, reduction='none') * existence_mask).sum(1).mean(0)
                existence_labels.append(existence_label[:, :test_dataset.max_objs])
                existence_probs.append(existence_prob[:, :test_dataset.max_objs])
        existence_labels = torch.cat(existence_labels, 0).cpu().detach().numpy()
        existence_probs = torch.cat(existence_probs, 0).cpu().detach().numpy()
        acc = acc/len(test_loader)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            avg_precision = average_precision_score(existence_labels, existence_probs)
        writer.add_scalar("charts/avg_precision_score", avg_precision, current_epoch)
        if args.track:
            writer.add_scalar("charts/test_coordinate_loss", acc, current_epoch)
            #wandb.log({'test_pred_y':predict_y}, commit=False)
            #wandb.log({'test_true_y':test_label})
        if not os.path.isdir("models"):
            os.mkdir("models")
        # torch.save(model, 'models/pongstack_{:.3f}.pkl'.format(acc/len(test_loader)))
        if avg_precision > best_avg_precision or acc < best_acc:
            print("new best!")
            if avg_precision > best_avg_precision:
                best_avg_precision = avg_precision
                print(f'epoch:{current_epoch} precision: {avg_precision}')
            if acc < best_acc:
                best_acc = acc
                print(f'epoch:{current_epoch} coordinate loss: {acc}')
            torch.save(model, 'models/'+f'{args.env_id}'+f'{args.resolution}'+f'{args.obj_vec_length}'+f"_gray{args.gray}"+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'_od.pkl')

        epoch_end_time = time.time()
        elapsed_time_per_epoch = epoch_end_time - epoch_start_time
        remaining_time = elapsed_time_per_epoch * (all_epoch - current_epoch - 1)
        #print(f"Estimated remaining time: {remaining_time/3600:.2f} hours")

    print(f"end best acc:{best_acc}")
    print("Model finished training")

#eval

    os.makedirs(os.path.join('stack_cnn_out_frames', args.env_id), exist_ok=True)
    batch_size = 1

    test_dataset = CustomImageDataset(images_dir_test,labels_test,args,train_flag=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = torch.load('models/'+f'{args.env_id}'+f'{args.resolution}'+f'{args.obj_vec_length}'+f"_gray{args.gray}"+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'_od.pkl')
    print(model)

    optm = Adam(model.parameters(), lr=args.lr)
    all_epoch = 1
    for current_epoch in range(all_epoch):
        model.eval()
        acc = 0
        img_path = os.path.join(images_dir, 'frame'+str(0)+'.png')
        image = Image.open(img_path).convert('RGB')
        size = image.size
        with torch.no_grad():
            progress_bar = tqdm(total=1000, desc=f"Processing Epoch {current_epoch}", position=0, leave=True)
            
            for idx, (test_x, test_label, _, _) in enumerate(test_loader):
                if idx > 999: 
                    break
                # global Img_size
                # Img_size = image.size
                if args.gray:
                    vis_obs = test_x[0,0,:,:]
                else:
                    vis_obs = test_x[0,0,:,:,:]
                vis_size = vis_obs.shape
                size = vis_size
                plt.figure(figsize=(10,10))
                plt.axis('off')
                if args.gray:
                    plt.imshow(vis_obs,plt.cm.gray)
                else:
                    plt.imshow(vis_obs)
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                predict_y = model(test_x.float(), threshold=args.threshold).detach()
                acc += coordinate_loss_fn(predict_y, test_label)
                # size = test_vis.shape
                # size = Img_size
                # test_vis = test_vis[:,:,0]
                # test_vis = cv2.cvtColor(test_vis, cv2.COLOR_GRAY2RGB)
                # plt.figure(figsize=(10,10))
                # plt.axis('off')
                # plt.imshow(test_vis)
                cors = test_label.cpu().numpy()
                pred_cors = predict_y.cpu().numpy()
                for i in range(args.n_objects):
                    cors[0][i*args.obj_vec_length] = cors[0][i*args.obj_vec_length]*size[0]
                    cors[0][i*args.obj_vec_length+1] = cors[0][i*args.obj_vec_length+1]*size[1]
                    pred_cors[0][i*args.obj_vec_length] = pred_cors[0][i*args.obj_vec_length]*size[0]
                    pred_cors[0][i*args.obj_vec_length+1] = pred_cors[0][i*args.obj_vec_length+1]*size[1]
                    plt.text(cors[0][i*args.obj_vec_length+1], cors[0][i*args.obj_vec_length], str(i+1), fontsize=36,color='red')
                    plt.text(pred_cors[0][i*args.obj_vec_length+1], pred_cors[0][i*args.obj_vec_length], str(i+1), fontsize=36,color='green')
                plt.savefig(os.path.join('stack_cnn_out_frames', args.env_id, f'{idx}.png'))
                plt.close()
                progress_bar.update(1) 
                
                current_loss = acc.item() / (idx + 1)
                progress_bar.set_description(f"Frame {idx} Loss: {current_loss:.4f}")
                
            progress_bar.close()
        
    print("Model finished test")
    
    input_path = os.path.join('stack_cnn_out_frames', args.env_id)
    output_path = input_path+'_seg.mp4'
    fps = 20 

    images_to_video(input_path, output_path, fps)
