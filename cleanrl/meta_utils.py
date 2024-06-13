import logging
import time

import cupy
import gymnasium as gym
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageStateObservation, ImageObservation
from metadrive.obs.state_obs import LidarStateObservation


def reconstruct_image_state(concatenated_image_state, state_shape, image_shape):
    state_length = np.prod(state_shape)
    images = concatenated_image_state[...,:-state_length]
    states = concatenated_image_state[...,-state_length:]
    if isinstance(concatenated_image_state, cupy.ndarray):
        array_module = cupy
    else:
        array_module = np
    if len(images.shape) == 1:
        # a single observation
        images = images.reshape(image_shape)
        stack_dim = len(images.shape) - 1
        images = array_module.transpose(
            images, (stack_dim,) + tuple(range(stack_dim)))
    elif len(images.shape) == 2:
        # has an extra batch dimension
        n_batch = concatenated_image_state.shape[0]
        images = images.reshape((n_batch,)+image_shape)
        stack_dim = len(images.shape) - 1
        images = array_module.transpose(
            images, (0,stack_dim) + tuple(range(1, stack_dim)))
    else:
        raise NotImplementedError
    images = images.astype(array_module.float32)
    states = states.astype(array_module.float32)
    return images, states

class ConcatenatedImageStateObservation(ImageStateObservation):
    """Flatten images and concatenate them with ego states."""
    def __init__(self, config):
        super().__init__(config)
        self.stack_size = config["stack_size"]
        self.image_shape = self.img_obs.observation_space.shape
        self.state_shape = self.state_obs.observation_space.shape
        
    @property
    def observation_space(self):
        full_length = np.prod(self.image_shape)+ np.prod(self.state_shape)
        return gym.spaces.Box(-1., 255, shape=(full_length,), dtype=np.float32)

    def observe(self, vehicle: BaseVehicle):
        image = self.img_obs.observe()
        if isinstance(image, cupy.ndarray):
            array_module = cupy
        else:
            array_module = np
        state = self.state_obs.observe(vehicle)
        state = array_module.array(state)
        image = image.flatten()
        image_state = array_module.concatenate([image, state], 0)
        return image_state.astype(array_module.float32)
            

class GrayScaledConcatenatedImageStateObservation(ConcatenatedImageStateObservation):
    """Flatten gray-scaled images and concatenate them with ego states."""
    def __init__(self, config):
        super().__init__(config)
        self.image_shape = self.img_obs.observation_space.shape[:2]\
            + (self.img_obs.observation_space.shape[-1], )

    def observe(self, vehicle: BaseVehicle):
        image = self.img_obs.observe()
        if isinstance(image, cupy.ndarray):
            array_module = cupy
        else:
            array_module = np
        image = array_module.transpose(image, (3,0,1,2))
        image = array_module.sum(
                array_module.multiply(
                    image, array_module.array(
                        [0.2125, 0.7154, 0.0721])), axis=-1)
        state = self.state_obs.observe(vehicle)
        state = array_module.array(state)
        image = image.flatten()
        image_state = array_module.concatenate([image, state], 0)
        return image_state.astype(array_module.float32)


class ConcatenatedImageLidarStateObservation(ConcatenatedImageStateObservation):
    """Flatten images and concatenate them with ego states and lidar point clouds."""
    def __init__(self, config):
        super(ImageStateObservation, self).__init__(config)
        self.img_obs = ImageObservation(
            config, config["vehicle_config"]["image_source"],
            config["norm_pixel"])
        self.state_obs = LidarStateObservation(config)
        self.image_shape = self.img_obs.observation_space.shape
        self.state_shape = self.state_obs.observation_space.shape


class GrayScaledConcatenatedImageLidarStateObservation(
    ConcatenatedImageLidarStateObservation,
    GrayScaledConcatenatedImageStateObservation):
    """Flatten gray-scaled images and concatenate them with ego states and lidar point clouds."""
    def __init__(self, config):
        super().__init__(config)
        self.image_shape = self.img_obs.observation_space.shape[:2]\
                + (self.img_obs.observation_space.shape[-1], )


class SubProcVecMetaDriveEnv(SubprocVecEnv): # TODO: check seeding
    def __init__(self, env_fns, args):
        super().__init__(env_fns)
        self.args = args
        # set action_space and observation_space to follow
        # the gym's vector env standard
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space
        self.n_envs = len(env_fns)
        low = float(self.action_space.low_repr)
        high = float(self.action_space.high_repr)
        self.action_space = gym.spaces.Box(
            low,
            high,
            shape=(self.n_envs,) + self.action_space.shape,
            dtype=np.float32)
        low = float(self.observation_space.low_repr)
        high = float(self.observation_space.high_repr)
        self.observation_space = gym.spaces.Box(
            low,
            high,
            shape=(self.n_envs,) + self.observation_space.shape,
            dtype=np.float32)
        if args.gray:
            self.image_shape = args.resolution, args.resolution, 4
            
        else:
            self.image_shape = args.resolution, args.resolution, 3, 4
        self.state_shape = self.single_observation_space.shape[0]\
            - np.prod(self.image_shape)


class DummySubProcVecMetaDriveEnv(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """For compatibility when using a single env."""
    def __init__(self, env_fns, args):
        assert len(env_fns) == 1
        env = env_fns[0]()
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self.args = args
        self.single_action_space = self.env.action_space
        self.single_observation_space = self.env.observation_space
        self.n_envs = 1
        low = float(self.action_space.low_repr)
        high = float(self.action_space.high_repr)
        self.action_space = gym.spaces.Box(
            low,
            high,
            shape=(self.n_envs,) + self.action_space.shape,
            dtype=np.float32)
        low = float(self.observation_space.low_repr)
        high = float(self.observation_space.high_repr)
        self.observation_space = gym.spaces.Box(
            low,
            high,
            shape=(self.n_envs,) + self.observation_space.shape,
            dtype=np.float32)
        if args.gray:
            self.image_shape = args.resolution, args.resolution, 4
        else:
            self.image_shape = args.resolution, args.resolution, 3, 4
        self.state_shape = self.single_observation_space.shape[0]\
            - np.prod(self.image_shape)
            
    def step(self, action):
        assert len(action.shape) == 2
        action = action[0]
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = np.array([reward])
        # convert to SB3 VecEnv api
        done = terminated or truncated
        info["TimeLimit.truncated"] = truncated and not terminated
        if done:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = observation
            observation = self.reset()
        else:
            observation = observation[None]
        return observation, reward, np.array([done]), [info]

    def reset(self, seed=None):
        observation, info = self.env.reset(seed=seed)
        return observation[None]


class RecordEpisodeStatistics(gym.wrappers.RecordEpisodeStatistics):
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

class ScaleReward(gym.RewardWrapper):
    def __init__(self, env, scaling_factor=1):
        super().__init__(env)
        self.scaling_factor = scaling_factor
    
    def reward(self, reward):
        return reward * self.scaling_factor

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs, info

def make_env(env_id, seed, idx, capture_video, run_name, resolution, gray=True, image_on_cuda=False, use_lidar=False, reward_scale=0.5):
    sensor_size = (resolution, resolution)
    if gray:
        if use_lidar:
            agent_observation = GrayScaledConcatenatedImageLidarStateObservation
        else:
            agent_observation = GrayScaledConcatenatedImageStateObservation
    else:
        if use_lidar:
            agent_observation = ConcatenatedImageLidarStateObservation
        else:
            agent_observation = ConcatenatedImageStateObservation
        
    cfg=dict(agent_observation=agent_observation,
         image_observation=True,
         horizon = 10000,
         norm_pixel=False,
         num_scenarios=1000,
         start_seed=1000, 
         vehicle_config=dict(image_source="rgb_camera"),
         sensors={"rgb_camera": (RGBCamera, *sensor_size)},
         stack_size=4,
         image_on_cuda=image_on_cuda,
         log_level=logging.CRITICAL)
    
    def thunk():
        if env_id == "MetaDriveEnv":
            env=MetaDriveEnv(cfg)
        else:
            raise NotImplementedError
        env = ScaleReward(env, reward_scale)
        env = RecordEpisodeStatistics(env)

        
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        set_random_seed(0)
        return env
    return thunk