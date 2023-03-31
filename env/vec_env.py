# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://arxiv.org/abs/2201.08355
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Vectorized environment wrapper.

This file contains a wrapper for vectorized environments. It is adapted from
codes from stable-baselines3 repository.
"""

import torch
import numpy as np
import pickle
import math
from tqdm import tqdm

from env.subproc_vec_env import SubprocVecEnv
from utils.misc import calculate_diversity, sample_with_range


def make_vec_envs(
    env_class,
    seed,
    num_envs,
    device,
    cpu_offset,
    cfg_env,
    dataset=None,
    num_gpu_envs=None,
    **kwargs,
):
    """
    Generate wrapped vectorized environments.

    Args:
        env_class (class): environment class
        seed (int): random seed
        num_envs (int): number of environments in data collection.
        device (torch.device): device to use for tensor operations
        cpu_offset (int): offset for CPU device id
        cfg_env (dict): environment configuration
        dataset (str): path to dataset
        num_gpu_envs (int): number of environments allowed on GPUs. Because we
            use dedicated policy for each environment when simulating
            trajectories, we need to limit the number of environments here,
            compared to num_envs.
        **kwargs: additional arguments for environment class
    """
    envs = [env_class(**kwargs) for _ in range(num_envs)]
    for rank, env in enumerate(envs):
        env.seed(seed + rank)
    if num_gpu_envs is None:
        num_gpu_envs = num_envs
    envs = VecEnvWrapper(
        envs, device, cpu_offset, cfg_env, num_gpu_envs, dataset
    )
    return envs


class VecEnvWrapper(SubprocVecEnv):

    def __init__(
        self,
        venv,
        device,
        cpu_offset,
        cfg_env,
        num_gpu_envs,
        dataset=None,
        start_method=None,
        pickle_option='cloudpickle',
    ):
        """
        Args:
            venv (VecEnv): vectorized environment
            device (torch.device): device to use for tensor operations
            cpu_offset (int): offset for CPU device id
            cfg_env (dict): environment configuration
            num_gpu_envs (int): number of environments allowed on GPUs. Because
                we use dedicated policy for each environment when simulating
                trajectories, we need to limit the number of environments here,
                compared to num_envs.
            dataset (str): path to dataset
            start_method (str): start method for multiprocessing
            pickle_option (str): option for pickle
        """
        super(VecEnvWrapper,
              self).__init__(venv, cpu_offset, start_method, pickle_option)
        self.device = device
        self.cfg_env = cfg_env
        self.num_gpu_envs = num_gpu_envs
        self.rng = np.random.default_rng(seed=cfg_env.venv_seed)

        # Sample for training envs - also validation?
        if dataset is not None:
            print("Load tasks from", dataset)
            with open(dataset, 'rb') as f:
                self.task_all = pickle.load(f)
            print(len(self.task_all), "tasks are loaded")
        else:
            print("Sample tasks")
            self.task_all = []
            for task_id in range(cfg_env.num_env_train):
                task = {}
                task['id'] = 'id_' + str(task_id)
                num_obs = self.rng.integers(
                    cfg_env.num_obs_range[0], cfg_env.num_obs_range[1],
                    endpoint=True
                )
                task['num_obs'] = num_obs
                obs_xs = sample_with_range(
                    cfg_env.obs_loc_x_range, num_obs, self.rng
                )
                obs_ys = sample_with_range(
                    cfg_env.obs_loc_y_range, num_obs, self.rng
                )
                task['obs_loc'] = np.concatenate(
                    (obs_xs[:, np.newaxis], obs_ys[:, np.newaxis]), axis=1
                )
                task['obs_radius'] = sample_with_range(
                    cfg_env.obs_radius_range, num_obs, self.rng
                )
                goal_y = sample_with_range(
                    cfg_env.goal_loc_y_range, 1, self.rng
                )[0]
                task['goal_loc'] = np.array([1.8, goal_y])
                self.task_all += [task]
        self.num_task = len(self.task_all)

    def sample_task(self, return_id=False):
        """
        Sample a task from the task pool.

        Args:
            return_id (bool): return task id

        Returns:
            task (dict): task information
        """
        task_id = self.rng.integers(0, self.cfg_env.num_env_train)
        if return_id:
            return self.task_all[task_id], task_id
        else:
            return self.task_all[task_id]

    def reset(
        self, use_default=False, random_init=False, state_init=None,
        verbose=False
    ):
        """
        Reset all environments.

        Args:
            use_default (bool): use default task
            random_init (bool): randomize initial state
            state_init (np.ndarray): initial state
            verbose (bool): print information

        Returns:
            obs (torch.Tensor): observation
            task_ids (np.ndarray or list): task ids
        """
        if use_default:
            task_all = [{} for _ in range(self.n_envs)]
            task_ids = np.arange(self.n_envs)
        else:
            task_info = [
                (self.sample_task(return_id=True)) for _ in range(self.n_envs)
            ]
            task_all = [info[0] for info in task_info]
            task_ids = [info[1] for info in task_info]
        args_all = [(random_init, state_init, task) for task in task_all]
        # obs = self.venv.reset_arg(args_all)
        obs = self.reset_arg(args_all)
        obs = torch.from_numpy(obs).to(self.device)
        if verbose:
            for index in range(self.n_envs):
                print("<-- Reset environment {}:".format(index))
                self.env_method(
                    # self.venv.env_method(
                    'report',
                    print_obs_state=False,
                    print_training=False,
                    indices=[index]
                )
        return obs, task_ids

    def reset_one(
        self, index, task=None, use_default=False, random_init=False,
        state_init=None, verbose=False
    ):
        """
        Reset one environment.

        Args:
            index (int): environment index
            task (dict): task
            use_default (bool): use default task
            random_init (bool): randomize initial state
            state_init (np.ndarray): initial state
            verbose (bool): print information

        Returns:
            obs (torch.Tensor): observation
            task_id (int): task id
        """
        task_id = -1  # dummy
        if task is None:
            if use_default:
                task = {}
            else:
                task, task_id = self.sample_task(return_id=True)
        obs = self.env_method(
            'reset', task=task, random_init=random_init, state_init=state_init,
            indices=[index]
        )[0]
        obs = torch.from_numpy(obs).to(self.device)
        if verbose:
            print("<-- Reset environment {}:".format(index))
            self.env_method(
                'report', print_obs_state=False, print_training=False,
                indices=[index]
            )
        return obs, task_id

    def get_append(self, states):
        """
        Get append information.

        Args:
            states (list): list of states

        Returns:
            append_all (torch.Tensor): append information
        """
        method_args_list = [(state,) for state in states]
        _append_all = self.env_method_arg(
            'get_append', method_args_list, indices=range(self.n_envs)
        )
        append_all = torch.FloatTensor(np.vstack(_append_all))
        return append_all.to(self.device)

    def get_obs(self, states):
        """
        Get observation.

        Args:
            states (list): list of states

        Returns:
            observations (torch.Tensor): observation
        """
        method_args_list = [(state,) for state in states]
        observations = torch.FloatTensor(
            self.env_method_arg(
                '_get_obs', method_args_list=method_args_list,
                indices=range(self.n_envs)
            )
        )
        return observations.to(self.device)

    def get_info(self, states):
        """
        Get info.

        Args:
            states (list): list of states

        Returns:
            info_all (list): list of info
        """
        method_args_list = [(state,) for state in states]
        info_all = self.env_method_arg(
            '_get_info', method_args_list, indices=range(self.n_envs)
        )
        return info_all

    def move_robot(self, actions, states):
        """
        Move robot.

        Args:
            actions (list): list of actions
            states (list): list of states

        Returns:
            states_new (list): list of new states
        """
        method_args_list = [
            (a_np, _state) for (a_np, _state) in zip(actions, states)
        ]
        states_new = self.env_method_arg(
            'move_robot', method_args_list, indices=range(self.n_envs)
        )
        return np.array(states_new)

    def step_async(self, actions):
        """Step asynchronously.

        In reality, the action space can be anything... - e.g., a trajectory
        plus the initial joint angles for the pushing task. We could also super
        this in each class to check the action space carefully.

        Args:
            actions (list): list of actions
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        super().step_async(actions)

    def step_wait(self):
        """
        Wait for step.

        Returns:
            obs (torch.Tensor): observation
            reward (torch.Tensor): reward
            done (torch.Tensor): done
            info (list): info
        """
        obs, reward, done, info = super().step_wait()
        obs = torch.from_numpy(obs).to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def simulate_trajectories(
        self,
        policy,
        mode,
        states=None,
        revert_task=False,
        sample_task=True,
        num_rnd_traj=1,
        # for sample_state
        sample_inside_obs=False,
        sample_inside_tar=False,

        # for simulate_one_trajectory
        T=None,
        end_type='TF',
        traj_size=0,
        frame_skip=0,
        latent_dist=None,
        latent=None,
        shield=False,
        backup=None,
        shield_dict=None,
    ):
        """
        Simulate trajectories.

        Args:
            policy (Policy): policy
            mode (str): mode
            states (np.ndarray): states
            revert_task (bool): revert task
            sample_task (bool): sample task
            num_rnd_traj (int): number of random trajectories
            sample_inside_obs (bool): sample inside obstacle
            sample_inside_tar (bool): sample inside target
            T (int): time horizon
            end_type (str): end type
            traj_size (int): trajectory size
            frame_skip (int): frame skip
            latent_dist (str): latent distribution
            latent (np.ndarray): latent
            shield (bool): shield
            backup (str): backup
            shield_dict (dict): shield dictionary

        Returns:
            length_all (list): list of traj length
            suc_all (list): list of traj success
            _state_final_all (list): list of final state
        """

        if states is not None:
            num_traj = states.shape[0]
        else:
            num_traj = num_rnd_traj
        num_process = self.num_gpu_envs
        num_batch = max(math.ceil(num_traj / num_process), 1)

        # For checking backup with RA, sample randomly in room env
        sample_random = False
        if mode == 'safety_ra' or mode == 'safety' or mode == 'risk':
            sample_random = True

        # keyword arguments for simulate_trajectories
        sample_args = dict(
            sample_inside_obs=sample_inside_obs,
            sample_inside_tar=sample_inside_tar, sample_random=sample_random
        )
        sim_traj_args = dict(
            T=T,
            end_type=end_type,
            traj_size=traj_size,
            frame_skip=frame_skip,
            latent_dist=latent_dist,
            latent=latent,
            shield=shield,
            backup=backup,
            shield_dict=shield_dict,
        )
        kwargs = dict(
            revert_task=revert_task,
            sample_args=sample_args,
            sim_traj_args=sim_traj_args,
        )

        suc_all = []
        length_all = []
        _state_final_all = []
        for batch_ind in tqdm(range(num_batch), leave=False):
            batch_idx = np.arange(
                batch_ind * num_process,
                min(num_traj, (batch_ind+1) * num_process)
            )
            method_args_list = []
            if states is not None:
                for state in states[batch_idx]:
                    tasks = None
                    if sample_task:
                        tasks = [self.sample_task()]
                    method_args_list += [
                        (policy, mode, state[np.newaxis], 1, tasks)
                    ]
            else:
                for _ in batch_idx:
                    tasks = None
                    if sample_task:
                        tasks = [self.sample_task()]
                    method_args_list += [(policy, mode, None, 1, tasks)]
            res_all = self.env_method_arg(
                'simulate_trajectories', method_args_list,
                indices=range(len(batch_idx)), **kwargs
            )
            suc_all += [res[1][0] for res in res_all]
            length_all += [len(res[0][0]) for res in res_all]
            _state_final_all += [res[0][0][-1, :] for res in res_all]
        return (
            np.array(length_all), np.array(suc_all),
            np.array(_state_final_all)
        )

    def simulate_all_envs(
        self,
        policy,
        mode,
        states=None,
        revert_task=False,
        num_traj_per_env=1,
        # for initial states
        sample_inside_obs=False,
        sample_inside_tar=False,
        fixed_init=False,
        # for simulate_one_trajectory
        T=None,
        end_type='TF',
        latent_dist=None,
        latent=None,
        shield=False,
        backup=None,
        shield_dict=None,
        traj_size=0,
        frame_skip=0,
    ):
        """
        For each environment, we sample `num_traj_per_env` latent variables
        with other keywords defined below.

        Args:
            policy (object): usually performance agent's actor.
            num_traj_per_env (int, optional): # of zs tested. Defaults to 1.
            state (np.ndarray, optional): initial state. Defaults to None.
            revert_task (bool, optional): after simulating, revert the task
                back if True. Defaults to False.
            end_type (str, optional): the condition to stop evaluation.
                Defaults to 'TF'.
            latent_dist_backup (object, optional): the method to sample latent
                variables for the backup agent. Defaults to None.
            latent_dist_policy (object, optional): the method to sample latent
                variables for the main agent. Defaults to None.
            T (int, optional): maximum steps in an episode. Defaults to None.
            sample_inside_obs (bool, optional): if state is not provided,
                sample the initial state also inside the obstacles if True.
                Defaults to False.
            sample_inside_tar (bool, optional):  if state is not provided,
                sample the initial state also inside the target if True.
                Defaults to False.
            shield (bool, optional): apply shielding during the rollout if
                True. Defaults to False.
            backup (object, optional): backup agent, containing both critic and
                agent. Defaults to None.
            shield_dict (dict, optional): specify the shielding type and the
                hyper-parameter for that shielding method. Defaults to None.

        Returns:
            (list of steps, list of results)
        """
        num_traj = self.num_task * num_traj_per_env
        num_batch = math.ceil(num_traj / self.n_envs)

        # For checking backup with RA, sample randomly in room env
        sample_random = False
        if mode == 'safety_ra' or mode == 'safety' or mode == 'risk':
            sample_random = True

        # keyword arguments for simulate_trajectories
        sample_args = dict(
            sample_inside_obs=sample_inside_obs,
            sample_inside_tar=sample_inside_tar, sample_random=sample_random,
            fixed_init=fixed_init
        )
        sim_traj_args = dict(
            T=T,
            end_type=end_type,
            latent_dist=latent_dist,
            latent=latent,
            shield=shield,
            backup=backup,
            shield_dict=shield_dict,
            traj_size=traj_size,
            frame_skip=frame_skip,
        )
        kwargs = dict(
            revert_task=revert_task,
            sample_args=sample_args,
            sim_traj_args=sim_traj_args,
        )

        if states is not None and states.ndim == 1:  # only one state
            states = np.tile(states[np.newaxis, :], (num_traj, 1))

        suc_all = []
        length_all = []
        for batch_ind in tqdm(range(num_batch), leave=False):
            batch_idx = np.arange(
                batch_ind * self.n_envs,
                min(num_traj, (batch_ind+1) * self.n_envs)
            )
            method_args_list = []
            if states is not None:
                for idx in batch_idx:
                    task_id = int(idx % len(self.task_all))
                    method_args_list += [(
                        policy, mode, states[idx][np.newaxis], 1,
                        [self.task_all[task_id]]
                    )]
            else:
                for idx in batch_idx:
                    task_id = int(idx % len(self.task_all))
                    method_args_list += [
                        (policy, mode, None, 1, [self.task_all[task_id]])
                    ]
            res_all = self.env_method_arg(
                'simulate_trajectories', method_args_list,
                indices=range(len(batch_idx)), **kwargs
            )
            suc_all += [res[1][0] for res in res_all]
            length_all += [len(res[0][0]) for res in res_all]
        return np.array(length_all), np.array(suc_all)

    # right now only deal with multiple latent variables of the main agent
    def simulate_multiple_latent_vars(
        self,
        policy,
        mode,
        task_id,
        num_traj_per_env=1,
        revert_task=False,
        # for initial states
        state=None,
        use_predefined=False,
        fixed_init=True,
        sample_inside_obs=False,
        sample_inside_tar=False,

        # for simulate_one_trajectory
        T=None,
        end_type='TF',
        latent_dist=None,
        latent=None,
        shield=False,
        backup=None,
        shield_dict=None,
        traj_size=0,
        frame_skip=0,
        num_grids_x=100,
        num_grids_y=100,
    ):
        if latent is not None:
            num_traj = latent.shape[0]
        else:
            num_traj = num_traj_per_env
        num_batch = math.ceil(num_traj / self.n_envs)

        sample_args = dict(
            sample_inside_obs=sample_inside_obs,
            sample_inside_tar=sample_inside_tar,
            fixed_init=fixed_init,
        )
        sim_traj_args = dict(
            T=T,
            end_type=end_type,
            traj_size=traj_size,
            frame_skip=frame_skip,
            latent_dist=latent_dist,
            shield=shield,
            backup=backup,
            shield_dict=shield_dict,
        )
        kwargs = dict(
            revert_task=revert_task,
            sample_args=sample_args,
            sim_traj_args=sim_traj_args,
        )

        if (state is None) and use_predefined:
            x_range = [0.1, 0.1]
            y_range = [-0.5, 0.5]
            theta_range = [-np.pi / 3, np.pi / 3]
            sample_x = self.rng.uniform(x_range[0], x_range[1], (num_traj, 1))
            sample_y = self.rng.uniform(y_range[0], y_range[1], (num_traj, 1))
            sample_theta = self.rng.uniform(
                theta_range[0], theta_range[1], (num_traj, 1)
            )
            sample_states = np.concatenate((sample_x, sample_y, sample_theta),
                                           axis=1)

        task = self.task_all[task_id]

        suc_all = []
        length_all = []
        trajectories = []
        for batch_ind in range(num_batch):
            print("[{}/{}]".format(batch_ind + 1, num_batch), end='\r')
            batch_idx = np.arange(
                batch_ind * self.n_envs,
                min(num_traj, (batch_ind+1) * self.n_envs)
            )
            method_args_list = []
            for i in batch_idx:
                if state is not None:
                    sample_state = [state]
                else:
                    if use_predefined:
                        sample_state = [sample_states[i]]
                    else:  # this uses env.sample_state(**sample_args)
                        sample_state = None

                method_args_list += [(policy, mode, sample_state, 1, [task])]
            res_all = self.env_method_arg(
                'simulate_trajectories', method_args_list,
                indices=range(len(batch_idx)), **kwargs
            )
            # The second [0] is due to the fact that env.simulate_trajectories
            # returns a list.
            suc_all += [res[1][0] for res in res_all]
            length_all += [len(res[0][0]) for res in res_all]
            trajectories += [res[0][0] for res in res_all]
        trajectories = np.array(trajectories, dtype=object)
        bounds = self.get_attr('bounds', indices=[0])[0]
        diversity = calculate_diversity(
            trajectories, bounds, num_grids_x, num_grids_y
        )[0]
        return np.array(length_all), np.array(suc_all), diversity

    def simulate_backup_all_envs(
        self, policy, mode, num_traj=1, state=None, task_id=0,
        revert_task=False, end_type='fail', traj_size=0, frame_skip=0,
        latent=None, latent_dist=None, T=None, sample_inside_obs=False,
        sample_inside_tar=False
    ):
        """
        For each environment, we sample `num_traj` initial states with other
        keywords defined below.

        Args:
            policy (object): usually performance agent's actor.
            mode (str): if the mode is not "performance", the neural network
                does not take in append.
            num_traj (int, optional): # of zs tested. Defaults to 1.
            state (np.ndarray, optional): initial state. Defaults to None.
            revert_task (bool, optional): after simulating, revert the task
                back if True. Defaults to False.
            end_type (str, optional): the condition to stop evaluation.
                Defaults to 'TF'.
            latent_dist_policy (object, optional): the method to sample latent
                variables for the policy agent. Defaults to None.
            T (int, optional): maximum steps in an episode. Defaults to None.
            sample_inside_obs (bool, optional): if state is not provided,
                sample the initial state also inside the obstacles if True.
                Defaults to False.
            sample_inside_tar (bool, optional):  if state is not provided,
                sample the initial state also inside the target if True.
                Defaults to False.

        Returns:
            (list of steps, list of results)
        """
        num_batch = math.ceil(num_traj / self.n_envs)

        sample_args = dict(
            sample_inside_obs=sample_inside_obs,
            sample_inside_tar=sample_inside_tar
        )
        sim_traj_args = dict(
            T=T, end_type=end_type, traj_size=traj_size, frame_skip=frame_skip,
            latent_dist=latent_dist
        )
        kwargs = dict(
            revert_task=revert_task,
            sample_args=sample_args,
            sim_traj_args=sim_traj_args,
        )

        if state is not None:
            state = [state]

        task = self.task_all[task_id]

        suc_all = []
        for batch_ind in range(num_batch):
            print("[{}/{}]".format(batch_ind, num_batch), end='\r')
            batch_idx = np.arange(
                batch_ind * self.n_envs,
                min(num_traj, (batch_ind+1) * self.n_envs)
            )
            method_args_list = []
            for _ in batch_idx:
                method_args_list += [(policy, mode, state, 1, [task])]
            res_all = self.env_method_arg(
                'simulate_trajectories', method_args_list,
                indices=range(len(batch_idx)), **kwargs
            )
            suc_all += [res[1][0] for res in res_all]
        return np.array(suc_all)
