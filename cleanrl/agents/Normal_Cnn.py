from typing import Tuple, TypeVar
from torch.nn import Module
from torch import nn
import torch
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_flat_size_after_conv(input_shape, conv_layers):
    size = torch.tensor(input_shape, dtype=torch.float32)
    for layer in conv_layers:
        if isinstance(layer, nn.Conv2d):
            size[0] = layer.out_channels
            size[1] = (size[1] - layer.kernel_size[0]) // layer.stride[0] + 1
            size[2] = (size[2] - layer.kernel_size[1]) // layer.stride[1] + 1
    return int(size[0] * size[1] * size[2])

def create_conv_layers(input_channels):
    return nn.Sequential(
        layer_init(nn.Conv2d(input_channels, 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
    )

def create_linear_layers(flat_size, output_size):
    return nn.Sequential(
        layer_init(nn.Linear(flat_size, 256)),
        nn.ReLU(),
        layer_init(nn.Linear(256, output_size))
    )

class BaseOD(nn.Module):
    def __init__(self, resolution, linear_output_size):
        super().__init__()
        
        self.conv_layers = create_conv_layers(resolution[0])
        flat_size = get_flat_size_after_conv(resolution, self.conv_layers)
        self.linear_layers = create_linear_layers(flat_size, linear_output_size)
        self.network = nn.Sequential(self.conv_layers, self.linear_layers)

    def forward(self, x):
        x = self.network(x)
        x = torch.clamp(x,0,1)
        return x

class OD_frames_gray(BaseOD):
    def __init__(self, args):
        self.obj_num = args.n_objects
        self.resolution = (4*1,args.resolution,args.resolution)
        super().__init__(self.resolution, self.obj_num*args.obj_vec_length*4)

class OD_frames(BaseOD):
    def __init__(self, args):
        self.obj_num = args.n_objects
        self.resolution = (4*3,args.resolution,args.resolution)
        super().__init__(self.resolution, self.obj_num*args.obj_vec_length*4)
    
    def forward(self, x):
        # state original shape: n_batch, n_frame, height, width, n_channel
        # convert to: n_batch, n_frame, n_channel, height, width
        x = torch.permute(x, (0, 1, 4, 2, 3))
        batch_size, n_frame, n_channels, height, width = x.shape
        x = x.reshape((batch_size, n_channels* n_frame, height, width))
        x = self.network(x)
        x = torch.clamp(x,0,1)
        return x

# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class Impala_CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._input_shape = (3, args.resolution, args.resolution)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(self._input_shape, out_channels)
            self._input_shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=self._input_shape[0] * self._input_shape[1] * self._input_shape[2], out_features=args.cnn_out_dim),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, x):
        x = x.permute((0, 3, 1, 2)) # "bhwc" -> "bchw"
        return self.network(x)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if self.use_norm:
            self.weight = nn.Parameter(torch.ones(out_channels))
            self.bias = nn.Parameter(torch.zeros(out_channels))
    def forward(self, x):
        x = self.m(x)
        if self.use_norm:
            return F.relu(F.group_norm(x, 1, self.weight, self.bias))
        else:
            return F.relu(x)


class Encoder(nn.Module):
    '''Encode an image to low-dimensional vectors using Conv2d.
    '''
    def __init__(self, img_channels, channels: Tuple[int, ...], strides: Tuple[int, ...], kernel_size):
        super().__init__()
        modules = []
        channel = img_channels
        for ch, s in zip(channels, strides):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channel, ch, kernel_size, stride=s, padding=kernel_size//2),
                    nn.ReLU(inplace=True),
                )
            )
            channel = ch
        self.conv = nn.Sequential(*modules)
    
    def forward(self, x):
        """
        input:
            x: input image, [B, img_channels, H, W]
        output:
            feature_map: [B, C, H_enc, W_enc]
        """
        x = self.conv(x)
        return x

class OD_frames_gray2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.obj_num = args.n_objects
        self.obj_vec_length = args.obj_vec_length
        self.n_frame = 4
        self.resolution = (self.n_frame*1, args.resolution, args.resolution)

        conv_layers = Encoder(self.resolution[0], [32, 64, 64], [2,2,1], 5)
        flat_size = get_flat_size_after_conv(self.resolution, conv_layers.conv)
        self.encoder = nn.Sequential(
            conv_layers,
            nn.Flatten(),
            nn.Linear(flat_size, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048))
        self.existence_layer = nn.Sequential(
            nn.Linear(2048, self.obj_num*self.n_frame),
            nn.ReLU(),
            nn.Linear(self.obj_num*self.n_frame, self.obj_num*self.n_frame))
        self.coordinate_layer = nn.Sequential(
            nn.Linear(2048, self.obj_num*args.obj_vec_length*self.n_frame),
            nn.ReLU(),
            nn.Linear(self.obj_num*args.obj_vec_length*self.n_frame, self.obj_num*args.obj_vec_length*self.n_frame))
        self.shape_layer = nn.Sequential(
            nn.Linear(2048, self.obj_num*2),
            nn.ReLU(),
            nn.Linear(self.obj_num*2, self.obj_num*2))

    def forward(self, x, return_existence_logits=False, clip_coordinates=True, return_shape=False, threshold=0):
        hidden = self.encoder(x)
        batch_size = x.shape[0]
        existence_logits = self.existence_layer(hidden)
        coordinates = self.coordinate_layer(hidden)
        shape = self.shape_layer(hidden).repeat(1, self.n_frame)
        if threshold > 0:
            coordinates = coordinates.reshape((
            batch_size, self.obj_num*self.n_frame, -1))
            existence_prob = torch.sigmoid(existence_logits)[:, :, None]
            coordinates = (coordinates * (existence_prob > threshold).float()).flatten(start_dim=1)
        if clip_coordinates:
            coordinates = torch.clamp(coordinates, 0, 1)
        if return_existence_logits and return_shape:
            return coordinates, existence_logits, shape
        elif return_existence_logits:
            return coordinates, existence_logits
        else:
            return coordinates