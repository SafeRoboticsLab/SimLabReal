# modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail 
import gym
import torch
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
import numpy as np
import math


def make_env(env_id, seed, rank, **kwargs):
    def _thunk():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)

        # hack - use same number of steps for train/eval
        # env = TimeLimitMask(env)

        # if log_dir is not None:
        #     env = Monitor(env,
        #                   os.path.join(log_dir, str(rank)),
        #                   allow_early_resets=allow_early_resets)

        # if len(env.observation_space.shape) == 3:
        #     raise NotImplementedError(
        #         "CNN models work only for atari,\n"
        #         "please use a custom wrapper for a custom pixel input env.\n"
        #         "See wrap_deepmind for an example.")

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  device,
                  **kwargs):
    envs = [
        make_env(env_name, seed, i, **kwargs)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # if len(envs.observation_space.shape) == 1:
    #     if gamma is None:
    #         envs = VecNormalize(envs, norm_reward=False)
    #     else:
    #         envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    # if num_frame_stack is not None:
    #     envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    # elif len(envs.observation_space.shape) == 3:
    #     envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
# class TimeLimitMask(gym.Wrapper):
#     def step(self, action):
#         obs, rew, done, info = self.env.step(action)
#         if done and self.env._max_episode_steps == self.env._elapsed_steps:
#             info['bad_transition'] = True

#         return obs, rew, done, info

#     def reset(self, **kwargs):
#         print(kwargs)
#         return self.env.reset(kwargs)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
    
    def simulate_trajectories(self, policy, states, endType, latent_prior):
        num_traj = states.shape[0]
        num_batch = math.ceil(num_traj/self.num_envs)

        suc_all = []
        for batch_ind in range(num_batch):
            batch_idx = np.arange(batch_ind*self.num_envs, min(num_traj, (batch_ind+1)*self.num_envs))
            method_args_list = []
            for state in states[batch_idx]:
                method_args_list += [(policy, state[np.newaxis], endType, latent_prior)]
            res_all = self.venv.env_method_arg('simulate_trajectories', method_args_list, indices=range(len(batch_idx)))
            suc_all += [res[1][0] for res in res_all]
        return np.array(suc_all)
