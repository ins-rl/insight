# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari.py
import argparse
import os
import random

from distutils.util import strtobool
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

#dataset
from torch.utils.data import DataLoader
from train_cnn import CustomImageDataset, coordinate_label_to_existence_label
from torch.nn import CrossEntropyLoss

import sympy as sym
from copy import copy
import torch
import numpy as np
from agents.eql.regularization import L12Smooth
from agents.eql.benchmark_l12 import var_names
from agents.eql import functions
import re

games_settings = {
    "PongNoFrameskip-v4": "Pong", 
    "BeamRiderNoFrameskip-v4": "BeamRider", 
    "EnduroNoFrameskip-v4": "Enduro",
    "QbertNoFrameskip-v4": "Qbert", 
    "SpaceInvadersNoFrameskip-v4": "SpaceInvaders", 
    "SeaquestNoFrameskip-v4": "Seaquest", 
    "BreakoutNoFrameskip-v4": "Breakout", 
    "FreewayNoFrameskip-v4": "Freeway", 
    "MsPacmanNoFrameskip-v4": "MsPacman", 
}


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="nsrl-eval",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--run-name", type=str, default=None,
        help="the defined run_name")
    parser.add_argument("--reg_weight", type=float, default=0,
        help="regulization for interpertable")
    parser.add_argument("--use_nn", type=lambda x: bool(strtobool(x)), default=True,
        help="use nn for critic")
    parser.add_argument("--cnn_out_dim", type=int, default=128,
        help="cnn_out_dim")
    parser.add_argument("--deter_action", type=lambda x: bool(strtobool(x)), default=False,
        help="deterministic action or not")
    parser.add_argument("--pre_nn_agent", type=lambda x: bool(strtobool(x)), default=False,
        help="load nn agent or not")
    parser.add_argument("--fix_cri", type=lambda x: bool(strtobool(x)), default=False,
        help="fix cri or not")
    parser.add_argument("--n_funcs", type=int, default=4,
        help="n_funcs")
    parser.add_argument("--n_layers", type=int, default=1,
        help="n_layers")
    parser.add_argument("--load_cnn", type=lambda x: bool(strtobool(x)), default=True,
        help="load_cnn")
    parser.add_argument("--cover_cnn", type=lambda x: bool(strtobool(x)), default=False,
        help="load_cnn and cover loaded neural agent")
    parser.add_argument("--ng", type=lambda x: bool(strtobool(x)), default=False,
        help="neural guided or not")
    parser.add_argument("--fix_cnn", type=lambda x: bool(strtobool(x)), default=False,
        help="fix_cnn")
    parser.add_argument("--visual", type=lambda x: bool(strtobool(x)), default=False,
        help="visualize or not")
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=True,
        help="save")
    parser.add_argument("--cnn_lr_drop", type=int, default=1,
        help="cnn_lr")
    parser.add_argument("--sam_track_data", type=lambda x: bool(strtobool(x)), default=True,
        help="use dataset generated by sam_track to train the agent")
    parser.add_argument("--mass_centri_cnn", type=bool, default=False,
        help="use mass_centri_cnn or not")
    parser.add_argument("--n_objects", type=int, default=256,
        help="n_objects")
    parser.add_argument("--resolution", type=int, default=84,
        help="resolution")
    parser.add_argument("--single_frame", type=bool, default=False,
        help="single frame or not")
    parser.add_argument("--cors", type=lambda x: bool(strtobool(x)), default=True,
        help="use cors")
    parser.add_argument("--bbox", type=lambda x: bool(strtobool(x)), default=False,
        help="use bbox")
    parser.add_argument("--rgb", type=lambda x: bool(strtobool(x)), default=False,
        help="use rgb")
    parser.add_argument("--obj_vec_length", type=int, default=2,
        help="obj vector length")
    parser.add_argument("--pre_train", type=lambda x: bool(strtobool(x)), default=False,
        help="pretrain agent or not")
    parser.add_argument("--pre_train_uptates", type=int, default=500,
        help="number of pre-train update")
    parser.add_argument("--gray", type=lambda x: bool(strtobool(x)), default=True,
        help="use gray or not")
    parser.add_argument("--clip_drop", type=lambda x: bool(strtobool(x)), default=False,
        help="drop clip-coef or not")
    parser.add_argument("--pnn_guide", type=lambda x: bool(strtobool(x)), default=False,
        help="use pure nn guide or not")
    parser.add_argument("--cnn_loss_weight", type=float, default=2.)
    parser.add_argument("--coordinate_loss", type=str, default="l1")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cnn_weight_decay", type=float, default=1e-4)
    parser.add_argument("--distillation_loss_weight", type=float, default=1)
    parser.add_argument("--reg_weight_drop", type=lambda x: bool(strtobool(x)), default=True,
        help="drop reg weight or not")   
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.obj_vec_length = args.cors*2+args.bbox*4+args.rgb*3
    args.cnn_out_dim = args.n_objects*args.obj_vec_length*4
    # fmt: on
    return args

def filter_mat(mat, threshold=0.01):
    """Remove elements of a matrix below a threshold."""
    mat[abs(mat)<threshold] = 0
    return mat

def apply_activation(W, funcs, n_double=0):
    """Given an (n, m) matrix W and (m) vector of funcs, apply funcs to W.

    Arguments:
        W:  (n, m) matrix
        funcs: list of activation functions (SymPy functions)
        n_double:   Number of activation functions that take in 2 inputs

    Returns:
        SymPy matrix with 1 column that represents the output of applying the activation functions.
    """
    W = sym.Matrix(W)
    if n_double == 0:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = funcs[j](W[i, j])
    else:
        W_new = W.copy()
        out_size = len(funcs)
        for i in range(W.shape[0]):
            in_j = 0
            out_j = 0
            while out_j < out_size - n_double:
                W_new[i, out_j] = funcs[out_j](W[i, in_j])
                in_j += 1
                out_j += 1
            while out_j < out_size:
                W_new[i, out_j] = funcs[out_j](W[i, in_j], W[i, in_j+1])
                in_j += 2
                out_j += 1
        for i in range(n_double):
            W_new.col_del(-1)
        W = W_new
    return W

def sym_pp(symbolic_layer, funcs, var_names, threshold=0.01, n_double=0):
    """Pretty print the hidden layers (not the last layer) of the symbolic regression network

    Arguments:
        funcs:  list of lambda functions using sympy. has the same size as W_list[i][j, :]
        var_names: list of strings for names of variables
        threshold: threshold for filtering expression. set to 0 for no filtering.
        n_double:   Number of activation functions that take in 2 inputs

    Returns:
        Simplified sympy expression.
    """
    vars = []
    n_double = functions.count_double(funcs)
    funcs = [func.sp for func in funcs]
    symbolic_layer = symbolic_layer.cpu()
    for var in var_names:
        if isinstance(var, str):
            vars.append(sym.Symbol(var))
        else:
            vars.append(var)
    expr = sym.Matrix(vars).T
    # W_list = np.asarray(W_list)
    weight = filter_mat(symbolic_layer.transform.weight.clone().cpu().detach().numpy(), threshold=threshold)
    weight = sym.Matrix(weight)
    bias = filter_mat(symbolic_layer.transform.bias.clone().cpu().detach().numpy(), threshold=threshold)
    bias = sym.Matrix(bias)
    expr = expr * weight.T + bias.T
    expr = apply_activation(expr, funcs, n_double=n_double)
    return expr

def last_pp(eq, linear_layer,  threshold=0.01):
    """Pretty print the last layer."""
    weight = filter_mat(linear_layer.weight.clone().cpu().detach().numpy(), threshold=threshold)
    weight = sym.Matrix(weight)
    bias = filter_mat(linear_layer.bias.clone().cpu().detach().numpy(), threshold=threshold)
    bias = sym.Matrix(bias)
    return (eq * weight.T + bias.T)

def extract_variable_numbers(expressions):
    variable_pattern = r'x(\d+)'
    variable_numbers = set()

    for expr in expressions:
        expr_str = str(expr)
        matches = re.findall(variable_pattern, expr_str)
        variable_numbers.update([int(match) for match in matches])

    return sorted(variable_numbers)

def make_env(env_id, seed, idx, capture_video, run_name,args):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (args.resolution, args.resolution))
        if args.gray:
            env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def eval_policy(envs, action_func, device="cuda", n_episode=10):
    obs,_ = envs.reset()
    episode_return = 0
    episode_length = 0
    total_episode = 0
    while total_episode<n_episode:
        obs_tensor = torch.Tensor(obs).to(device)
        with torch.no_grad():
            action = action_func(obs_tensor)
            obs, _, _, _, info = envs.step(action.cpu().numpy())
        if "final_info" in info:
            for env_id, env_info in enumerate(info["final_info"]):
                if not env_info is None:
                    if "episode" in env_info:
                        episode_return += env_info["episode"]["r"]
                        episode_length += env_info["episode"]["l"]
                        total_episode += 1.
                        if total_episode == n_episode:
                            break
    return episode_return / total_episode, episode_length / total_episode



if __name__ == "__main__":
    args = parse_args()
    if args.run_name == None:
        run_name = f"{args.env_id}"+f'_{args.obj_vec_length}'+f"_gray{args.gray}"+f"_t{args.total_timesteps}"
        if args.pre_nn_agent:
            run_name+="_pre_nn_agent"
        if args.ng:
            run_name+="_ng"
        if args.pnn_guide:
            run_name+="_png"
        if args.fix_cnn:
            run_name+="_fix_cnn"
        if args.cover_cnn:
            run_name+="_cover_cnn"
        run_name += f"_objs{args.n_objects}"
        run_name+=f"_seed{args.seed}"
    else:
        run_name = args.run_name
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name='Coor-weight-wong-remain',
            monitor_gym=True,
            save_code=True,
        )
        table = wandb.Table(columns=["Environment ID", "Mean Acc", "Std Acc"])



    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.coordinate_loss == "l2":
        coordinate_loss_fn = torch.nn.functional.mse_loss
    elif args.coordinate_loss == "l1":
        coordinate_loss_fn = torch.nn.functional.l1_loss
    else:
        raise NotImplementedError
    regularization = L12Smooth()
    loss_distill = CrossEntropyLoss()
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name,args) for i in range(args.num_envs)])
    envs_eval = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name,args) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    accs_data = {}
    for env_seed in range(1,4):
        args.seed = env_seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        for id in games_settings:
            #sam_track_data:
            asset_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "cleanrl/sam_track/assets")
            images_dir = os.path.join(asset_dir, f'{id}'+'_masks_train')
            labels = os.path.join(images_dir, 'labels.json')
            images_dir_test = os.path.join(asset_dir, f'{id}'+'_masks_test')
            labels_test = os.path.join(images_dir_test, 'labels.json')

            test_dataset = CustomImageDataset(images_dir_test,labels_test,args,train_flag=True)
            test_loader = DataLoader(test_dataset, batch_size=1)
            test_data = iter(test_loader)
            print(id,args.seed)
            agent = torch.load(f'cleanrl/models/agents/yours.pth').to(device)

            #extract most important coor
            n_variables = 2048
            expre = sym_pp(agent.eql_actor.layers[0],  agent.activation_funcs, var_names[:n_variables], threshold=1e-2) # 1e-1 for general explanation
            expre = last_pp(expre, agent.eql_actor.layers[1], threshold=1e-2)

            extracted_variables_indice = extract_variable_numbers(expre)
            exist_obs = set(int(i/2) for i in extracted_variables_indice)


            acc = 0
            agent.network.eval()
            with torch.no_grad():
                for idx, (test_x, test_label, _, _) in enumerate(test_loader):
                    test_x = test_x.to(device)
                    test_label = test_label.to(device)
                    existence_label, existence_mask = coordinate_label_to_existence_label(test_label)
                    for i in range(1024):
                        if i not in exist_obs:
                            existence_label[0,i] = 0
                            existence_mask[0,i*2] = 0
                            existence_mask[0,i*2+1] = 0
                    # existence_label[0,i] = 0 for i not in extracted_variables_indice
                    existence_label_0 =torch.count_nonzero(existence_label[0,:256])
                    existence_label_1 =torch.count_nonzero(existence_label[0,256:512])
                    existence_label_2 =torch.count_nonzero(existence_label[0,512:768])
                    existence_label_3 =torch.count_nonzero(existence_label[0,768:1024])
                    # print(existence_label[0,1536:2048])
                    existence_label_nonzero = torch.zeros(1, 2048).to(device)
                    existence_label_nonzero[0, :512] = 1 / existence_label_0 if existence_label_0 > 0 else 0
                    existence_label_nonzero[0, 512:1024] = 1 / existence_label_1 if existence_label_1 > 0 else 0
                    existence_label_nonzero[0, 1024:1536] = 1 / existence_label_2 if existence_label_2 > 0 else 0
                    existence_label_nonzero[0, 1536:2048] = 1 / existence_label_3 if existence_label_3 > 0 else 0
                    non_zero_frame = torch.count_nonzero(torch.tensor([existence_label_0,existence_label_1,existence_label_2,existence_label_3]))
                    predict_y = agent.network(test_x.float(), threshold=0.5).detach()
                    acc = acc + (coordinate_loss_fn(predict_y, test_label, reduction='none') * existence_mask*existence_label_nonzero).sum(1).mean(0)/non_zero_frame/2
            print(f"losses/test_cnn_dataset_loss {acc/len(test_loader)}")
            if args.track:
                mean_acc = acc/len(test_loader)
                key = (id)
                if key not in accs_data:
                    accs_data[key] = []
                accs_data[key].append(mean_acc.detach().cpu())
                table.add_data(id, round(np.mean(accs_data[key])*100, 1),round(np.std(accs_data[key])*100, 1))
                wandb.log({"Accs": copy(table)})
    writer.close()
    wandb.finish()
