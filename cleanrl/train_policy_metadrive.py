# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
os.environ['PYTHONUTF8'] = '1' #to enable utf8
import random
import time
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from agents.eql.regularization import L12Smooth
from agents.agent import AgentContinues
from itertools import chain

#dataset
from torch.utils.data import DataLoader
from train_cnn import CustomImageDataset, coordinate_label_to_existence_label, binary_focal_loss_with_logits
from torch.nn import CrossEntropyLoss

from visualize_utils import visual_for_agent_videos
from tqdm import tqdm
import itertools
from collections import deque

from stable_baselines3.common.utils import set_random_seed
from meta_utils import reconstruct_image_state, SubProcVecMetaDriveEnv, make_env

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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL-Metadrive",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MetaDriveEnv",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
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
    parser.add_argument("--kl-penalty-coef", type=float, default=0,
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
    parser.add_argument("--resolution", type=int, default=128,
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
    parser.add_argument("--use_eql_actor", type=lambda x: bool(strtobool(x)), default=False,
        help="use eql actor to collect data or not")   
    parser.add_argument("--deter_eql", type=lambda x: bool(strtobool(x)), default=True,
        help="eql deter action for distill or not")   
    #metadrive
    parser.add_argument("--ego_state", type=lambda x: bool(strtobool(x)), default=False,
        help="use ego_state or not")
    parser.add_argument("--ego_state_dim", type=int, default=19,
        help="metadrive ego-state dim")
    parser.add_argument("--lidar", type=lambda x: bool(strtobool(x)), default=False,
        help="use lidar or not")
    parser.add_argument("--reward_scale", type=float, default=0.1,
        help="the scaling factor for environmental rewards")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.obj_vec_length = args.cors*2+args.bbox*4+args.rgb*3
    args.cnn_out_dim = args.n_objects*args.obj_vec_length*4
    # fmt: on
    return args

def eval_policy(envs, action_func, device="cuda", n_episode=10):
    obs = envs.reset()
    obs, state = reconstruct_image_state(
    obs, state_shape=envs.state_shape, image_shape=envs.image_shape)
    episode_return = 0
    episode_length = 0
    total_episode = 0
    success_episode = 0
    while total_episode<n_episode:
        obs_tensor = torch.Tensor(obs).to(device)
        state_tensor = torch.Tensor(state).to(device)
        with torch.no_grad():
            action = action_func(obs_tensor,state_tensor)
            obs, _, _,  info = envs.step(action.cpu().numpy()) #for sb3, we only have 4 values
            obs, state = reconstruct_image_state(
            obs, state_shape=envs.state_shape, image_shape=envs.image_shape)
        for item in info:
            if "episode" in item.keys():
                total_episode += 1
                episode_return += item["episode"]["r"]
                episode_length += item["episode"]["l"]
                if item['arrive_dest']:
                    success_episode+=1
                if total_episode == n_episode:
                    break
    return episode_return / total_episode, episode_length / total_episode, success_episode/ total_episode



if __name__ == "__main__":
    args = parse_args()
    if args.run_name == None:
        run_name = f"{args.env_id}"+f'_{args.obj_vec_length}'+f"_gray{args.gray}"+f"_ego{args.ego_state}"+f"_t{args.total_timesteps}"
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
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    #sam_track_data:
    asset_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "cleanrl/sam_track/assets")
    images_dir = os.path.join(asset_dir, f'{args.env_id}'+'_masks_train')
    labels = os.path.join(images_dir, 'labels.json')
    images_dir_test = os.path.join(asset_dir, f'{args.env_id}'+'_masks_test')
    labels_test = os.path.join(images_dir_test, 'labels.json')

    train_dataset = CustomImageDataset(images_dir,labels,args,train_flag=True)
    test_dataset = CustomImageDataset(images_dir_test,labels_test,args,train_flag=True)
    train_loader = DataLoader(train_dataset, batch_size=args.minibatch_size,num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.minibatch_size)
    train_data = itertools.cycle(train_loader)
    test_data = iter(test_loader)


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
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = SubProcVecMetaDriveEnv(
        [make_env(args.env_id, args.seed + i, i, capture_video=False, run_name='', resolution=args.resolution, gray=True,reward_scale=args.reward_scale, use_lidar=args.lidar) for i in range(args.num_envs)], args)
    envs_eval = SubProcVecMetaDriveEnv(
        [make_env(args.env_id, args.seed + i, i, capture_video=False, run_name='', resolution=args.resolution, gray=True,reward_scale=args.reward_scale, use_lidar=args.lidar) for i in range(args.num_envs)], args)
    args.ego_state_dim = np.prod(envs.state_shape)

    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name,args) for i in range(args.num_envs)])
    # envs_eval = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name,args) for i in range(args.num_envs)])
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    if args.pre_nn_agent:
        if args.pnn_guide:
            print('pnn_guide')
            agent_nn = torch.load('models/agents/'+f"{args.env_id}"+f'_{args.obj_vec_length}'+"_NN"+"_gray"*args.gray+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'.pth').to(device)
        else:
            agent_nn = torch.load('models/agents/'+f"{args.env_id}"+f'_{args.obj_vec_length}'+f"_gray{args.gray}"+f"_t{args.total_timesteps}"+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'.pth').to(device)
            print('nn_guide')
        agent = AgentContinues(envs,args,agent_nn).to(device)
        for param in agent_nn.parameters():
            param.requires_grad = False
        if args.cover_cnn:
            agent.network = torch.load('models/'+f'{args.env_id}'+f'{args.resolution}'+f'{args.obj_vec_length}'+f"_gray{args.gray}"+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'_od.pkl')
    else:
        agent = AgentContinues(envs,args).to(device)

    #fix hypara
    if args.fix_cnn:
            for param in agent.network.parameters():
                param.requires_grad = False
    if args.fix_cri:
            for param in agent.critic.parameters():
                param.requires_grad = False
    actor_params = chain(
        agent.neural_actor.parameters(),
        agent.eql_actor.parameters(),
        [agent.eql_actor_logstd, agent.neural_actor_logstd])
    optimizer = optim.Adam(
        [{'params':actor_params,'lr':args.learning_rate },
         {'params':agent.critic.parameters(),'lr':args.learning_rate},
         {'params':agent.network.parameters(),'lr':args.learning_rate/args.cnn_lr_drop,'weight_decay': args.cnn_weight_decay}], eps=1e-5)
    
    #sam_track_data:
    # cnn_optm = Adam(agent.network.parameters(), lr=args.learning_rate)
    if args.coordinate_loss == "l2":
        coordinate_loss_fn = torch.nn.functional.mse_loss
    elif args.coordinate_loss == "l1":
        coordinate_loss_fn = torch.nn.functional.l1_loss
    else:
        raise NotImplementedError
    # cnn_model = agent.network.to(device)
    regularization = L12Smooth()
    loss_distill = nn.MSELoss(reduction='none')
    # ALGO Logic: Storage setup
    env_img_shape = (envs.image_shape[2],envs.image_shape[0],envs.image_shape[1])
    obs = torch.zeros((args.num_steps, args.num_envs) + env_img_shape).to(device) #for img shape
    states = torch.zeros((args.num_steps, args.num_envs) + (envs.state_shape,)).to(device) #for img shape
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs, next_state = reconstruct_image_state(
        next_obs, state_shape=envs.state_shape, image_shape=envs.image_shape)
    next_obs = torch.as_tensor(next_obs, device=device)
    next_state = torch.as_tensor(next_state, device=device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    success_deque = deque(maxlen=10)

    # Setup directory for visualization
    os.makedirs('ppoeql_stack_cnn_out_frames', exist_ok=True)
    os.makedirs(os.path.join('ppoeql_stack_cnn_out_frames', run_name), exist_ok=True)
    os.makedirs(os.path.join('ppoeql_stack_cnn_out_frames', run_name,'test'), exist_ok=True)
    os.makedirs(os.path.join('ppoeql_stack_cnn_out_frames', run_name,'record'), exist_ok=True)

    with torch.no_grad():
        # cnn loss  
        agent.network.train()
        # t1 = time.time()


    with tqdm(total=num_updates, desc="Training Progress") as pbar:
        for update in range(1, num_updates + 1):
            u_rate = update/num_updates
            if update%int(num_updates/100) ==0 or update==1:
                acc = 0
                agent.network.eval()
                with torch.no_grad():
                    for idx, (test_x, test_label, _, _) in enumerate(test_loader):
                        test_x = test_x.to(device)
                        test_label = test_label.to(device)
                        existence_label, existence_mask = coordinate_label_to_existence_label(test_label)
                        predict_y = agent.network(test_x.float(), threshold=0.5).detach()
                        acc = acc + (coordinate_loss_fn(predict_y, test_label, reduction='none') * existence_mask).sum(1).mean(0)
                if args.ng:
                    action_func = lambda t,s: agent.get_action_and_value(
                        t,
                        threshold=args.threshold,
                        actor="eql",
                        next_state=s,
                        deterministic=args.deter_eql)[0]
                    eql_returns, eql_lengths, eql_success_rate = eval_policy(
                        envs_eval, action_func, device=device)
                    writer.add_scalar(
                        "charts/eql_returns", eql_returns, global_step)
                    writer.add_scalar(
                        "charts/eql_lengths", eql_lengths, global_step)
                    writer.add_scalar(
                        "charts/eql_success_rate", eql_success_rate, global_step)
                agent.network.train()
                writer.add_scalar("losses/test_cnn_dataset_loss", acc/len(test_loader), global_step)
            if update%int(num_updates/10) ==0 or update==1:
                """visual_for_agent_videos(envs_eval, agent, next_obs, device, args,run_name, threshold=args.threshold, next_state=next_state)
                video_path = os.path.join('ppoeql_stack_cnn_out_frames', run_name, 'test_seg.mp4')
                wandb.log({"test_seg": wandb.Video(video_path, fps=20, format="mp4")})"""
                if args.save:
                    torch.save(agent, 'models/agents/'+run_name+'.pth')

            # Annealing
            frac = 1.0 - (update - 1.0) / num_updates
            if args.anneal_lr:
                lrnow = frac * args.learning_rate
            else:
                lrnow = args.learning_rate
            if args.clip_drop:
                clip_coef_now = frac * args.clip_coef
            else:
                clip_coef_now = args.clip_coef
            if args.reg_weight_drop:
                # 还可以先为0一段时间，然后逐渐增加
                completed_ratio = (update - 1.0) / num_updates
                reg_weight_now = args.reg_weight * completed_ratio
            else:
                reg_weight_now = args.reg_weight
                # optimizer.param_groups[0]["lr"] = lrnow
            optimizer.param_groups[0]["lr"] = lrnow
            optimizer.param_groups[1]["lr"] = lrnow
            optimizer.param_groups[2]["lr"] = lrnow/args.cnn_lr_drop

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                states[step] = next_state
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    if args.pre_train and update<args.pre_train_uptates:
                        action, logprob, _, value,_,_ = agent.get_pretrained_action_and_value(next_obs,next_state)
                    else:
                        if args.use_eql_actor:
                            action, logprob, _, value,_,_ = agent.get_action_and_value(next_obs, threshold=args.threshold, actor='eql', next_state=next_state)
                        else:
                            action, logprob, _, value,_,_ = agent.get_action_and_value(next_obs, threshold=args.threshold,next_state=next_state)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob


                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done,  info = envs.step(action.cpu().numpy())
                next_obs, next_state = reconstruct_image_state(
                next_obs, state_shape=envs.state_shape, image_shape=envs.image_shape)
                next_obs = torch.as_tensor(next_obs, device=device)
                next_state = torch.as_tensor(next_state, device=device)
                rewards[step] = torch.as_tensor(reward, device=device).view(-1)
                next_done = torch.as_tensor(
                    done, device=device, dtype=torch.float32)
                for item in info:
                    if "episode" in item.keys():
                        # print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        success_deque.append(item['arrive_dest'])
                        writer.add_scalar("charts/episodic_success_rate", sum(success_deque)/len(success_deque), global_step)
                        break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs,next_state).reshape(1, -1)
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + env_img_shape)
            b_states = states.reshape((-1,) + (envs.state_shape,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_actions = b_actions.float() #for suqashed gaussian
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            writer.add_scalar("charts/adv_mean", b_advantages.mean(), global_step)
            writer.add_scalar("charts/adv_std", b_advantages.std(), global_step)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    # import time
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    if args.use_eql_actor:
                        _, newlogprob, entropy, newvalue,new_action_mean,newprob = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], threshold=args.threshold, actor='eql', next_state=b_states[mb_inds])
                    else:
                        _, newlogprob, entropy, newvalue,new_action_mean,newprob = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], threshold=args.threshold,next_state=b_states[mb_inds])
                    _, _, _, _, eql_action_mean, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], threshold=args.threshold, actor="eql",next_state=b_states[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    approx_kl = ((ratio - 1) - logratio).mean()
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        
                        clipfracs += [((ratio - 1.0).abs() > clip_coef_now).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef_now, 1 + clip_coef_now)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -clip_coef_now,
                            clip_coef_now,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    if epoch == args.update_epochs - 1:
                        if args.ng:
                            #distillation_loss = loss_distill(
                            #    eq_logits, b_actions[mb_inds])
                            distillation_loss = loss_distill(
                                eql_action_mean,
                                new_action_mean)
                            distillation_loss = distillation_loss.sum(1).mean(0)
                            reg_loss = regularization(agent.eql_actor.get_weights_tensor())
                            writer.add_scalar("losses/reg_policy_loss", reg_loss, global_step)
                            writer.add_scalar("losses/distill_policy_loss", distillation_loss, global_step)
                        else:
                            distillation_loss = 0
                            reg_loss = 0
                        if args.cnn_loss_weight > 0:
                            # t1 = time.time()
                            train_x, train_label, train_label_weight, train_shape = next(train_data)
                            # t2 = time.time()
                            train_x = train_x.to(device)
                            train_label = train_label.to(device)
                            train_label_weight = train_label_weight.to(device)
                            train_shape = train_shape.to(device)
                            existence_label, existence_mask = coordinate_label_to_existence_label(train_label)
                            train_label_weight_mask = train_label_weight.unsqueeze(-1).repeat(1, 1, args.obj_vec_length).flatten(start_dim=1)
                            predict_y, existence_logits, predict_shape = agent.network(
                                train_x.float(), return_existence_logits=True, clip_coordinates=False, return_shape=True)
                            coordinate_loss = (coordinate_loss_fn(predict_y, train_label, reduction='none') * existence_mask * train_label_weight_mask).sum(1).mean(0)
                            shape_loss = (coordinate_loss_fn(predict_shape, train_shape, reduction='none') * existence_mask * train_label_weight_mask).sum(1).mean(0)
                            existence_loss = binary_focal_loss_with_logits(existence_logits, existence_label, reduction='none')
                            existence_loss = (existence_loss * train_label_weight).sum(1).mean(0)
                            loss_cnn = coordinate_loss + existence_loss + shape_loss
                            writer.add_scalar("losses/cnn_dataset_loss", loss_cnn, global_step)
                        else:
                            loss_cnn = 0
                    else:
                        distillation_loss = 0
                        reg_loss = 0
                        loss_cnn = 0
                    loss = pg_loss - args.ent_coef * entropy_loss\
                           + args.kl_penalty_coef * approx_kl \
                           + args.vf_coef * v_loss\
                           + args.cnn_loss_weight * loss_cnn\
                           + args.distillation_loss_weight * distillation_loss\
                           + reg_weight_now * reg_loss
                    optimizer.zero_grad()
                    loss.backward()
                    if args.max_grad_norm > 0:
                        grad_norm = nn.utils.clip_grad_norm_(
                            agent.parameters(), args.max_grad_norm)
                    else:
                        grad_norm = 0
                    optimizer.step()
                    writer.add_scalar("charts/ppo_grad_norm", grad_norm, global_step)
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break
    
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            
            writer.add_scalar("charts/update", update, global_step)

            writer.add_scalar("charts/action0", b_actions.mean(0)[0], global_step)
            writer.add_scalar("charts/action1", b_actions.mean(0)[1], global_step)
            pbar.update(1)   
    if args.save:
        torch.save(agent, 'models/agents/'+run_name+'.pth')
    #eql
    '''if args.eql:
        with torch.no_grad():
            expra = pretty_print.network(agent.actor.get_weights(), agent.activation_funcs, var_names[:args.cnn_out_dim])
            for i in range(envs.single_action_space.n):
                print(f"action{i}:")
                sy.pprint(sy.simplify(expra[i]))'''
    # visual_for_videos(envs, agent.network, next_obs,device,args,run_name)
    envs.close()
    writer.close()
