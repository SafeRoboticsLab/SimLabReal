# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://arxiv.org/abs/2201.08355
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Base Environment

Parent class for both Vanilla and Advanced environments

Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
         Allen Z. Ren (allen.ren@princeton.edu)
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from collections import deque
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt

from utils.misc import check_shielding, get_frames, rgba2rgb


class BaseEnv(ABC):

    def __init__(
        self,
        task=None,
        render=False,
        use_rgb=True,
        img_h=128,
        img_w=128,
        use_append=True,
        obs_buffer=0.05,
        boundary_buffer=0.0,
        g_x_fail=1.0,
        num_traj_per_visual_init=1,
        fixed_init=False,
        sparse_reward=False,
        sample_inside_obs=False,
        sample_inside_tar=True,
        max_step_train=100,
        max_step_eval=100,
        done_type='fail',
        terminal_type='const',
        reward_type='all',
        reward_goal=5.,
        reward_obs=-5.,
        reward_wander=2.,
    ):
        """
        Args:
            task (str): Task name
            render (bool): Whether to render the environment
            use_rgb (bool): Whether to use RGB image
            img_h (int): Image height
            img_w (int): Image width
            use_append (bool): Whether to additional info to neural network
            obs_buffer (float): Buffer for obstacles; added to safety margin
            boundary_buffer (float): Buffer for room boundaries; added to safety margin
            g_x_fail (float): distance to obstacle if the robot is inside the obstacle - for HJ calculation purpose
            num_traj_per_visual_init (int): Number of trajectories for each visualization
            fixed_init (bool): Whether to use fixed initial state
            sparse_reward (bool): Whether to use sparse reward
            sample_inside_obs (bool): Whether to sample the initial state inside obstacles
            sample_inside_tar (bool): Whether to sample the initial state inside the target
            max_step_train (int): Maximum number of steps for training
            max_step_eval (int): Maximum number of steps for evaluation
            done_type (str): Type of done
                - 'TF': terminate episode when robot succeeds or fails
                - 'fail': terminate episode when robot fails
                - 'end': terminate episode when robot reaches outside the boundary
            terminal_type (str): Type of terminal [TODO: KC]
            reward_type (str): Type of reward
                - 'all': reward for reaching the goal and penalty for hitting the obstacle
                - 'task': reward for reaching the goal
            reward_goal (float): Reward for reaching the goal
            reward_obs (float): Reward for hitting the obstacle
            reward_wander (float): Reward for wandering
        """
        super(BaseEnv, self).__init__()
        assert (reward_type == 'all' or reward_type == 'task'
                or reward_type == 'risk'),\
            'The reward type should be all, task or risk.'
        self.img_h = img_h
        self.img_w = img_w
        self.use_append = use_append
        self.obs_buffer = obs_buffer
        self.boundary_buffer = boundary_buffer
        self.g_x_fail = g_x_fail
        self.num_traj_per_visual_init = num_traj_per_visual_init
        self.fixed_init = fixed_init
        self.sparse_reward = sparse_reward
        self.use_rgb = use_rgb
        self.render = render
        self.sample_inside_obs = sample_inside_obs
        self.sample_inside_tar = sample_inside_tar
        self.max_step_train = max_step_train
        self.max_step_eval = max_step_eval
        self.done_type = done_type
        self.terminal_type = terminal_type
        self.reward_type = reward_type
        self.reward_goal = reward_goal
        self.reward_obs = reward_obs
        self.reward_wander = reward_wander

        # PyBullet ID
        self._physics_client_id = -1

        # Flag for train/eval
        self.set_train_mode()

        # Set up observation and action space for Gym
        if use_rgb:
            self.num_img_channel = 3  # RGB
        else:
            self.num_img_channel = 1  # D only
        self.observation_shape = (self.num_img_channel, img_h, img_w)
        self.camera_tilt_range = None
        self.camera_tilt_noise_std = 0
        self.camera_roll_noise_std = 0
        self.camera_yaw_noise_std = 0
        self._goal_yaw_range = np.array([[0, 2 * np.pi]])
        self.get_yaw_append = False

    @property
    def env_type(self):
        """
        Type of environment
        """
        raise NotImplementedError

    @property
    def state_dim(self):
        """
        Dimension of robot state
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self):
        """
        Dimension of robot action
        """
        raise NotImplementedError

    def set_train_mode(self):
        """
        Set the environment to train mode.
        """
        self.flag_train = True
        self.max_steps = self.max_step_train

    def set_eval_mode(self):
        """
        Set the environment to eval mode.
        """
        self.flag_train = False
        self.max_steps = self.max_step_eval

    def seed(self, seed=0):
        """
        Set the seed of the environment. Should be called after action_sapce is
        defined.

        Args:
            seed (int, optional): random seed value. Defaults to 0.
        """
        self.seed_val = seed
        # self.action_space.seed(self.seed_val)
        self.rng = np.random.default_rng(seed=self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(
            self.seed_val
        )  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # return [seed]

    @abstractmethod
    def init_pb(self):
        """
        Initialize PyBullet environment.
        """
        raise NotImplementedError

    @abstractmethod
    def close_pb(self):
        """
        Kills created obkects and closes pybullet simulator.
        """
        raise NotImplementedError

    @abstractmethod
    def set_default_task(self):
        """
        Set default task for the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_task(self):
        """
        Reset the task for the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_obstacles(self):
        """
        Reset obstacle for the environment. Called in reset() if loading
        obstacles for the 1st time, or in reset_task() if loading obstacles for
        the 2nd time.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_goal(self):
        """
        Reset goal for the environment. Called in reset() if loading goal for
        the 1st time, or in reset_task() if loading goal for the 2nd time.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_robot(self):
        """
        Reset robot for the environment. Called in reset() if loading robot for
        the 1st time, or in reset_task() if loading robot for the 2nd time.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, random_init=False, state_init=None, task=None):
        """
        Reset the environment, including robot state, task, and obstacles.
        Initialize pybullet client if 1st time.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self):
        """
        Step function. Apply action, move robot, get observation,
        calculate reward, check if done.
        """
        raise NotImplementedError

    @abstractmethod
    def safety_margin(self, return_boundary=False):
        """
        Compute safety margin.
        """
        raise NotImplementedError

    def get_target_yaw_margin(self, state):
        target_yaw_margin = float("inf")
        for tmp_range in self._goal_yaw_range:
            yaw_lb = tmp_range[0] - state[2]
            yaw_ub = state[2] - tmp_range[1]
            target_yaw_margin = min(target_yaw_margin, max(yaw_lb, yaw_ub))
        return target_yaw_margin

    def target_margin(self, state, return_yaw=False):
        """
        Compute the margin (e.g. distance) between state and target set.

        Args:
            state: consisting of [x, y, theta].

        Returns:
            float: negative value suggests inside of the
                target set.
        """
        s = state[:2]
        c_r = [self._goal_loc, self._goal_radius]
        target_dist_margin = self._calculate_margin_circle(
            s, c_r, negative_inside=True
        )
        if len(state) > 2:  # check theta
            target_yaw_margin = self.get_target_yaw_margin(state)
            target_margin = max(target_yaw_margin, target_dist_margin)
            if return_yaw:
                return target_margin, target_yaw_margin
            else:
                return target_margin
        else:
            return target_dist_margin

    @abstractmethod
    def move_robot(self):
        """
        Move robot to the next state; returns next state
        """
        raise NotImplementedError

    @abstractmethod
    def report(self):
        """
        Print information of robot dynamics and observation.
        """
        raise NotImplementedError

    @abstractmethod
    def visualize(self):
        """
        Visualize trajectories and value functions.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_state(self):
        """
        Sample a state from the state space.
        """
        raise NotImplementedError

    @staticmethod
    def _calculate_margin_rect(s, x_y_w_h_theta, negative_inside=True):
        """
        Calculate the margin to the box.

        Args:
            s (np.ndarray): the state.
            x_y_w_h_theta (np.ndarray): box specification, (center_x, center_y,
                width, height, tilting angle), angle is in rad, height and
                width are of half extent.
            negativeInside (bool, optional): add a negative sign to the
                distance if inside the box. Defaults to True.

        Returns:
            float: margin.
        """
        x, y, w, h, theta = x_y_w_h_theta
        delta_x = s[0] - x
        delta_y = s[1] - y
        delta_normal = np.abs(
            delta_x * np.sin(theta) - delta_y * np.cos(theta)
        )
        delta_tangent = np.sqrt(delta_x**2 + delta_y**2 - delta_normal**2)
        margin = max(delta_tangent - w, delta_normal - h)

        if negative_inside:
            return margin
        else:
            return -margin

    @staticmethod
    def _calculate_margin_circle(s, c_r, negative_inside=True):
        """
        Calculate the margin to the circle.

        Args:
            s (list): x/y state.
            c_r (list): x/y/r of the circle.
            negative_inside (bool, optional): whether to set negative value if
                inside circle.

        Returns:
            float: the margin to the circle.
        """
        center, radius = c_r
        dist_to_center = np.linalg.norm(s[:2] - center)
        margin = dist_to_center - radius
        if negative_inside:
            return margin
        else:
            return -margin

    def _get_info(self, state, action=None):
        """
        Get infortmation of the environment - mostly called in step().

        Returns
            dict: the information of the environment.
        """
        l_x, target_yaw_margin = self.target_margin(state, return_yaw=True)
        # normalized_lx = l_x / self._init_goal_dist
        g_x, boundary_margin, obstacle_margin = self.safety_margin(
            state, return_boundary=True
        )
        heading_vec = self._goal_loc - state[:2]
        heading = np.arctan2(heading_vec[1], heading_vec[0]) - state[2]
        if heading > np.pi:
            heading -= 2 * np.pi
        elif heading < -np.pi:
            heading += 2 * np.pi
        info = {
            'task': self._task,
            'state': state,
            'g_x': g_x,
            'boundary_margin': boundary_margin,
            'obstacle_margin': obstacle_margin,
            'l_x': l_x,
            "heading": heading,
            'target_yaw_margin': target_yaw_margin,
        }
        if action is not None:
            info['l_x_ra'] = np.linalg.norm(
                np.array(state[:2]) - self._init_state[:2]
            ) - 0.5
        return info

    def get_append(self, state):
        """
        Gets l_x and heading for append. Adds acceptable yaw range if asked.

        Returns:
            np.array: l_x, heading and yaw_append (optional).
        """
        if self.use_append:
            info = self._get_info(state)
            l_x = info['l_x']
            heading = info['heading']
            _append = np.array([l_x, heading])
        else:
            _append = np.array([0.0, 0.0])

        if self.get_yaw_append:
            _append = np.append(_append, self.yaw_append)

        return _append[np.newaxis]

    def _get_obs(self, state):
        """
        Get RGB or depth image given a state

        Args:
            state (np.ndarray): (x, y, yaw)

        Returns:
            np.ndarray: RGB or depth image, of the shape (C, H, W)
        """
        # State
        if len(state) == 3:  # car
            x, y, yaw = state
        else:  # spirit
            x, y, z, yaw = state[:4]

        # Add noise
        camera_roll_noise = 0
        camera_yaw_noise = 0
        if self.camera_tilt_range is not None:
            camera_tilt = np.random.uniform(
                self.camera_tilt_range[0], self.camera_tilt_range[1], 1
            )[0]
        else:
            camera_tilt = self.camera_tilt
            if self.camera_tilt_noise_std > 0:  # in deg
                camera_tilt += np.random.normal(
                    0, self.camera_tilt_noise_std, 1
                )[0]
        if self.camera_roll_noise_std > 0:  # in deg
            camera_roll_noise = np.random.normal(
                0, self.camera_roll_noise_std, 1
            )[0]
        if self.camera_yaw_noise_std > 0:  # in deg
            camera_yaw_noise = np.random.normal(
                0, self.camera_yaw_noise_std, 1
            )[0]

        #
        rot_matrix = [
            camera_roll_noise / 180 * np.pi, camera_tilt / 180 * np.pi,
            yaw + camera_yaw_noise / 180 * np.pi
        ]
        rot_matrix = self._p.getMatrixFromQuaternion(
            self._p.getQuaternionFromEuler(rot_matrix)
        )
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (1, 0, 0)  # x-axis
        init_up_vector = (0, 0, 1)  # z-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        cam_pos = np.array([x, y, self.camera_height]) + rot_matrix.dot(
            (self.camera_x_offset, 0, 0)
        )
        view_matrix = self._p.computeViewMatrix(
            cam_pos, cam_pos + 0.1*camera_vector, up_vector
        )

        # Get Image
        far = 1000.0
        near = 0.01
        projection_matrix = self._p.computeProjectionMatrixFOV(
            fov=self.camera_fov, aspect=self.camera_aspect, nearVal=near,
            farVal=far
        )
        _, _, rgb_img, depth, _ = self._p.getCameraImage(
            self.img_w, self.img_h, view_matrix, projection_matrix,
            flags=self._p.ER_NO_SEGMENTATION_MASK, shadow=1,
            lightDirection=[1, 1, 1]
        )
        depth = np.reshape(depth, (1, self.img_h, self.img_w))
        depth = far * near / (far - (far-near) * depth)
        if self.use_rgb:
            rgb = rgba2rgb(rgb_img).transpose(2, 0, 1)  # store as uint8
            # rgb = rgba2rgb(rgb_img).transpose(2, 0, 1) / 255  # CHW
            return rgb
        else:
            return depth

    def get_axes(self):
        """
        Get bounds of the environments and true aspect ratio.

        Returns:
            axes (np.ndarray): the bounds of the environment.
            aspect_ratio (float): the true aspect ratio of the environment.
        """
        aspect_ratio = ((self.bounds[0, 1] - self.bounds[0, 0]) /
                        (self.bounds[1, 1] - self.bounds[1, 0]))
        axes = np.array([
            self.bounds[0, 0], self.bounds[0, 1], self.bounds[1, 0],
            self.bounds[1, 1]
        ])
        return [axes, aspect_ratio]

    def get_warmup_examples(self, num_warmup_samples=100):
        """
        Get warmup examples for the environment.

        Args:
            num_warmup_samples (int, optional): the number of warmup samples.

        Returns:
            states: the warmup states.
            heuristic_v: for safety Bellman equation.
        """
        heuristic_v = np.zeros((num_warmup_samples, 1))
        states = np.zeros(shape=(num_warmup_samples,) + self.observation_shape)

        for i in range(num_warmup_samples):
            _state = self.sample_state(
                self.sample_inside_obs, self.sample_inside_tar
            )
            l_x = self.target_margin(_state)
            g_x = self.safety_margin(_state)
            heuristic_v[i, :] = np.maximum(l_x, g_x)
            states[i] = self._get_obs(_state)

        return states, heuristic_v

    def get_value(
        self,
        q_func,
        theta,
        nx=101,
        ny=101,
        batch_size=128,
        latent=None,
        traj_size=0,
    ):
        """
        Get the state values given the Q-network. We fix the heading
            angle of the car to `theta`.

        Args:
            q_func (object): agent's Q-network.
            theta (float): the heading angle of the car.
            nx (int, optional): # points in x-axis. Defaults to 101.
            ny (int, optional): # points in y-axis. Defaults to 101.

        Returns:
            np.ndarray: values
        """
        v = np.zeros((nx, ny))
        it = np.nditer(v, flags=['multi_index'])
        xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
        ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
        stack_frame = traj_size > 0

        while not it.finished:
            # getCameraImage somehow hangs at the walls
            # if (
            #     abs(x) == self.state_bound or abs(x) == 0
            #     or abs(y) == self.state_bound / 2
            # ):
            #     v[idx] = 0
            # else:
            append_all = []
            obs_all = np.empty((0, 3, self.img_h, self.img_w), dtype='uint8')
            idx_all = []
            g_x_all = []
            for _ in range(batch_size):
                idx = it.multi_index
                idx_all += [idx]
                x = xs[idx[0]]
                y = ys[idx[1]]

                state = np.array([x, y, theta])
                g_x_all.append(self.safety_margin(state))
                obs = self._get_obs(state)
                obs_all = np.concatenate((obs_all, obs[np.newaxis]))
                append_all += [self.get_append(state)]
                it.iternext()
                if it.finished:
                    break
            append_all = np.concatenate(append_all)
            # not specifying latent here; q_func() will use latent mean if latent is used
            if stack_frame:
                obs_stack_all = np.tile(
                    obs_all, (1, traj_size, 1, 1)
                )  # repeat obs traj_size times
                v_all = q_func(
                    obs=obs_stack_all,
                    append=append_all,
                )
            else:
                v_all = q_func(obs_all, append=append_all)
            for v_s, idx, g_x in zip(v_all, idx_all, g_x_all):
                if g_x > 0:
                    v[idx] = self.g_x_fail
                else:
                    v[idx] = v_s
        return v, xs, ys

    def check_within_bounds(self, state):
        """
        Check if robot is within bounds

        Args:
            state (np.ndarray): (x, y, yaw)

        Returns:
            bool: True if inside the environment.
        """
        for dim, bound in enumerate(self.bounds):
            flag_low = state[dim] < bound[0]
            flag_high = state[dim] > bound[1]
            if flag_low or flag_high:
                return False
        return True

    # == Trajectories Rollout ==
    def simulate_one_trajectory(
        self, policy, mode, state, T=None, end_type='TF', traj_size=0,
        frame_skip=0, prev_obs=None, latent_dist=None, latent=None,
        shield=False, backup=None, shield_dict=None
    ):
        """
        Simulate the trajectory given an initial state and a policy.

        Args:
            policy (func): agent.
            mode (str): if the mode is not "performance", the neural network
                does not take in append.
            state (np.ndarray, optional): if provided, set the initial state to
                its value. Defaults to None.
            T (int, optional): the maximum length of the trajectory.
                Defaults to 250.
            end_type (str, optional): when to end the traj. Defaults to 'TF'.

        Returns:
            np.ndarray: states of the trajectory, of the shape (length, 3).
            int: result.
            float: the minimum reach-avoid value of the trajectory.
            dictionary: extra information, (v_x, g_x, l_x, obs) along the
                trajectory.
        """
        _state = state

        result = 0  # not finished
        traj = []
        observations = []
        value_list = []
        gx_list = []
        lx_list = []

        # Check if mixed stack - only in shielding
        mixed_stack = False
        if backup is not None and backup.obs_channel == 3:
            mixed_stack = True

        # Sample latent for trajectory
        context_perf = None
        context_backup = None

        if latent is not None:
            context_perf = latent.clone()
            context_perf = context_perf.to(policy.device)
        elif latent_dist is not None:
            context_perf = latent_dist.sample().view(1, -1)
            context_perf = context_perf.to(policy.device)

        if backup is not None:
            if backup.has_latent:
                context_backup = backup.latent_dist.sample((1,))
                context_backup = context_backup.to(backup.device)

        # Initialize prev_obs
        if prev_obs is None:
            prev_obs = deque(maxlen=traj_size)  # empty is fine; can be one

        # Set shielding
        if shield:
            shield_type = shield_dict['type']
            shield_inst = []
        else:
            shield_type = "no"

        use_traj_fwd = ((shield_type == 'simulator')
                        or (shield_type == 'mixed'))
        if use_traj_fwd:
            traj_fwd_list = []

        # Set max steps
        if T is None:
            T = self.max_steps

        # Reset car before rollout
        self.reset_robot(_state)
        if end_type == 'safety_ra':
            action_converted = [1., 0]
        for t in range(T):

            # Get obs, g, l
            obs = self._get_obs(_state)[np.newaxis]  # 1x1xHxW
            traj.append(_state)
            observations.append(obs)
            if end_type == 'safety_ra':
                info = self._get_info(_state, action_converted)
                l_x_ra = info['l_x_ra']
            else:
                info = self._get_info(_state)
            l_x = info['l_x']
            g_x = info['g_x']

            # Update prev_obs
            obs_tensor = torch.from_numpy(obs).to(policy.device).squeeze(0)
            prev_obs.appendleft(obs_tensor)

            # Add rollout record
            if t == 0:
                max_g = g_x
                current = max(l_x, max_g)
                min_v = current
            else:
                max_g = max(max_g, g_x)
                current = max(l_x, max_g)
                min_v = min(current, min_v)
            value_list.append(min_v)
            gx_list.append(g_x)
            lx_list.append(l_x)

            # Check the termination criterion
            if end_type == 'end':
                done = not self.check_within_bounds(_state)
                if done:
                    result = -1
            elif end_type == 'TF':
                if g_x > 0:
                    result = -1  # failed
                    break
                elif l_x <= 0:
                    result = 1  # succeeded
                    break
            elif end_type == 'safety_ra':
                if g_x > 0:
                    result = -1  # failed
                    break
                elif l_x_ra <= 0:
                    result = 1  # succeeded
                    break
            elif end_type == 'fail':
                if g_x > 0:
                    result = -1  # failed
                    break

            # Get append
            append = self.get_append(_state)

            # Get the context/latent
            if traj_size > 0:  # simple frame stacking
                obs_perf = get_frames(prev_obs, traj_size,
                                      frame_skip).unsqueeze(0)
                if not mixed_stack:
                    obs_backup = obs_perf
                else:
                    obs_backup = obs
            else:
                obs_perf = obs
                obs_backup = obs

            # Infer policy
            action = policy.actor(obs_perf, append=append, latent=context_perf)

            # Add indicator - implicitly remove the leading dim
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            action = np.append(action, policy.act_ind.cpu())

            # Save robot state for reloading
            prev_state = np.copy(_state)

            # Run shielding mechanism
            if shield:
                # rej needs perf_obs - assume back_obs the same
                shield_flag, shield_info = check_shielding(
                    backup, shield_dict, obs_backup, action, append,
                    context_backup=context_backup, state=prev_state,
                    policy=policy, context_policy=context_perf
                )

                if shield_type == 'simulator' or shield_type == 'mixed':
                    traj_fwd_list.append(shield_info['traj_fwd'])

                # Shield action
                if shield_flag:
                    if shield_type == 'rej':
                        action = shield_info['action_final']  # always in np?
                        action = np.append(action, policy.act_ind.cpu())
                    else:
                        action = backup.actor(
                            obs_backup, append=append, latent=context_backup
                        )
                        if isinstance(action, torch.Tensor):
                            action = action.detach().cpu().numpy()
                        action = np.append(action, backup.act_ind.cpu())
                    shield_inst.append(True)
                else:
                    shield_inst.append(False)

                # Reset robot to the previous state before shielding
                if shield_type != 'value':
                    self.reset_robot(prev_state)

            # Forward dynamics
            if end_type == 'safety_ra':
                _state, action_converted = self.move_robot(
                    action, prev_state, return_converted=True
                )
            else:
                _state = self.move_robot(action, prev_state)

        # Save info
        traj = np.array(traj)
        observations = np.array(observations)
        info = {
            'value_list': value_list,
            'gx_list': gx_list,
            'lx_list': lx_list,
            'observations': observations
        }
        if use_traj_fwd:
            info['traj_fwd_list'] = traj_fwd_list
        if shield:
            shield_inst = np.array(shield_inst)
            info['shield_inst'] = shield_inst
        return traj, result, min_v, info

    def simulate_trajectories(
        self, policy, mode, states=None, num_rnd_traj=None, tasks=None,
        revert_task=False, sample_args=None, sim_traj_args=None, **kwargs
    ):
        """
        Simulate the trajectories. If the states are not provided, we pick the
        initial states from the discretized state space.

        Args:
            policy (func): agent's policy.
            mode (str): if the mode is not "performance", the neural network
                does not take in append.
            states (np.ndarray, optional): if provided, set the initial states
                to its value. Defaults to None.
            num_rnd_traj (int, optional): #states sampled for the initial state
                if states are not provided. Defaults to None.
            tasks (list, optional): list of dictionaries, each dictionary
                specifies a specific task. If it is not provided, use the
                current loaded task. Defaults to None.
            revert_task (bool, optional): revert to the current task after the
                simulation if True. Defaults to False.
            sample_args (dict, optional): arguments for sample_state().
                Defaults to None.
            sim_traj_args (dict, optional): arguments for
                simulate_one_trajectory(). Defaults to None.

        Returns:
            list of np.ndarray: each element is a tuple consisting of x and y
                positions along the trajectory.
            np.ndarray: the binary reach-avoid outcomes.
            np.ndarray: the minimum reach-avoid values of the trajectories.
        """
        assert ((num_rnd_traj is None and states is not None)
                or (num_rnd_traj is not None and states is None)
                or (len(states) == num_rnd_traj))
        if states is not None:
            num_rnd_traj = len(states)

        trajectories = []

        if sample_args is None:
            sample_args = {}
        if sim_traj_args is None:
            sim_traj_args = {}

        if revert_task:
            prev_task = self._task

        results = np.empty(shape=(num_rnd_traj,), dtype=int)
        min_vs = np.empty(shape=(num_rnd_traj,), dtype=float)
        infos = []
        for idx in range(num_rnd_traj):

            # Reset task if provided
            if tasks is not None:
                self.reset_task(tasks[idx])

            # Sample state if not provided
            if states is None:
                state = self.sample_state(**sample_args)
            else:
                state = states[idx]

            # Run simulation
            traj, result, min_v, info = self.simulate_one_trajectory(
                policy, mode, state=state, **sim_traj_args
            )
            trajectories.append(traj)
            results[idx] = result
            min_vs[idx] = min_v
            infos.append(info)

        if revert_task:
            self.reset_task(prev_task)

        return trajectories, results, min_vs, infos

    # == Plotting ==
    def plot_v_values(
        self,
        q_func,
        fig,
        ax,
        theta=np.pi / 2,
        bool_plot=False,
        plot_cbar=True,
        vmin=-1,
        vmax=1,
        nx=101,
        ny=101,
        cmap='seismic',
        normalize_v=False,
        alpha=1,
        plot_contour=True,
        fontsize=40,
        #
        latent=None,
        traj_size=0,
    ):
        """
        Plot values.

        Args:
            q_func (object): agent's Q-network.
            fig (matplotlib.figure)
            ax (matplotlib.axes.Axes)
            theta (float, optional): if provided, fix the car's heading angle
                to its value. Defaults to np.pi/2.
            bool_plot (bool, optional): plot the values in binary form.
                Defaults to False.
            plot_cbar (bool, optional): plot the color bar or not.
                Defaults to True.
            vmin (int, optional): vmin in colormap. Defaults to -1.
            vmax (int, optional): vmax in colormap. Defaults to 1.
            nx (int, optional): # points in x-axis. Defaults to 101.
            ny (int, optional): # points in y-axis. Defaults to 101.
            cmap (str, optional): color map. Defaults to 'seismic'.
        """
        ax_style = self.get_axes()

        # == Plot V ==
        if theta is None:
            theta = 2.0 * self.rng.uniform() * np.pi
        v, xs, ys = self.get_value(
            q_func, theta, nx, ny, latent=latent, traj_size=traj_size
        )

        if bool_plot:
            im = ax.imshow(
                v.T > 0., interpolation='none', extent=ax_style[0],
                origin="lower", cmap=cmap, zorder=-1
            )
        else:
            if normalize_v:
                vmin_label = np.min(v)
                vmax_label = np.max(v)
                vmean_label = (vmin_label+vmax_label) / 2
                v = (v - np.min(v)) / (np.max(v) - np.min(v))
                v = vmin + v * (vmax-vmin)
            im = ax.imshow(
                v.T, interpolation='none', extent=ax_style[0], origin="lower",
                cmap=cmap, vmin=vmin, vmax=vmax, zorder=-1, alpha=alpha
            )
            if plot_contour:
                ax.contour(
                    xs, ys, v.T, levels=[0], colors='g', linewidths=2,
                    linestyles='dashed'
                )
            if plot_cbar:
                if not normalize_v:
                    vmin_label = vmin
                    vmax_label = vmax
                    vmean_label = 0

                cbar = fig.colorbar(
                    im, ax=ax, pad=0.02, fraction=0.2, shrink=.95,
                    ticks=[vmin, 0, vmax]
                )
                v_ticklabels = np.around(
                    np.array([vmin_label, vmean_label, vmax_label]), 2
                )
                cbar.ax.set_yticklabels(labels=v_ticklabels, fontsize=fontsize)

    def plot_trajectories(
        self, policy, mode, ax, num_rnd_traj=None, states=None, c='k', lw=1,
        sz=1, zorder=2, plot_dot=True, print_step=False, sample_args=None,
        sim_traj_args=None, **kwargs
    ):
        """
        Plot trajectories given the agent's Q-network.

        Args:
            policy (func): agent's policy.
            ax (matplotlib.axes.Axes).
            num_rnd_traj (int, optional): Defaults to None.
            T (int, optional): the maximum length of the trajectory.
                Defaults to 250.
            end_type (str, optional): when to end the traj. Defaults to 'TF'.
            states (np.ndarray, optional): if provided, set the initial states
                to its value. Defaults to None.
            theta (float, optional): if provided, set the theta to its value.
                Defaults to np.pi/2.
            sample_inside_obs (bool, optional): sampling initial states inside
                of the obstacles or not. Defaults to True.
            sample_inside_tar (bool, optional): sampling initial states inside
                of the targets or not. Defaults to True.
            c (str, optional): color. Defaults to 'k'.
            lw (float, optional): lw. Defaults to 1.5.
            zorder (int, optional): graph layers order. Defaults to 2.

        Returns:
            np.ndarray: the binary reach-avoid outcomes.
            np.ndarray: the minimum reach-avoid values of the trajectories.
        """
        assert ((num_rnd_traj is None and states is not None)
                or (num_rnd_traj is not None and states is None)
                or (len(states) == num_rnd_traj))

        trajectories, results, min_vs, infos = self.simulate_trajectories(
            policy, mode, num_rnd_traj=num_rnd_traj, states=states,
            sample_args=sample_args, sim_traj_args=sim_traj_args, **kwargs
        )

        use_shielding = False
        if sim_traj_args is not None:
            if 'shield' in sim_traj_args:
                use_shielding = sim_traj_args['shield']

        if ax is None:
            ax = plt.gca()
        for traj, result, info in zip(trajectories, results, infos):

            if 'shield_inst' in info.keys():
                shield_inst = info['shield_inst']
            else:
                shield_inst = [False] * len(traj)
            traj_x = traj[:, 0]
            traj_y = traj[:, 1]

            # Plot init
            ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)

            if use_shielding:
                shield_inst = info['shield_inst']
            else:
                shield_inst = []
            if plot_dot:
                t = 0
                for inst in np.where(shield_inst)[0]:
                    ax.scatter(
                        traj_x[t:inst + 1], traj_y[t:inst + 1], s=sz, c=c,
                        zorder=zorder
                    )
                    ax.scatter(
                        traj_x[inst:inst + 2], traj_y[inst:inst + 2], s=sz,
                        c='g', zorder=zorder
                    )
                    t = inst + 1
                ax.scatter(
                    traj_x[t:-1], traj_y[t:-1], s=sz, c=c, zorder=zorder
                )
            else:
                t = 0
                for inst in np.where(shield_inst)[0]:
                    ax.plot(
                        traj_x[t:inst + 1], traj_y[t:inst + 1], c=c, ls='-',
                        lw=lw, zorder=zorder
                    )
                    ax.plot(
                        traj_x[inst:inst + 2], traj_y[inst:inst + 2], c='g',
                        lw=lw, zorder=zorder
                    )
                    t = inst + 1
                ax.plot(
                    traj_x[t:-1], traj_y[t:-1], c=c, ls='-', lw=lw,
                    zorder=zorder
                )

            # Mark failure or success/safe
            if result == -1:
                ax.scatter(traj_x[-1], traj_y[-1], s=24, c='r', zorder=zorder)
            elif result == 1:
                ax.scatter(traj_x[-1], traj_y[-1], s=24, c='g', zorder=zorder)
            else:
                ax.scatter(traj_x[-1], traj_y[-1], s=24, c='b', zorder=zorder)

        if print_step:
            num_steps = [len(traj) for traj in trajectories]
            ax.text(
                x=0, y=0, s=' '.join(str(step) for step in num_steps),
                fontsize=20, transform=plt.gcf().transFigure
            )
        return results, min_vs

    def plot_formatting(
        self, ax, labels=None, fsz=20, use_equal_aspect_ratio=True
    ):
        """
        Format plot.

        Args:
            ax (matplotlib.axes.Axes).
            labels (list, optional): x- and y- labels. Defaults to None.
            fsz (int, optional): font size. Defaults to 20.
        """
        ax_style = self.get_axes()
        # ax.plot([0., 0.], [ax_style[0][2], ax_style[0][3]], c='k')
        # ax.plot([ax_style[0][0], ax_style[0][1]], [0., 0.], c='k')
        ax.axis(ax_style[0])
        if use_equal_aspect_ratio:
            ax.set_aspect(ax_style[1])  # makes equal aspect ratio
        ax.grid(False)
        if labels is not None:
            ax.set_xlabel(labels[0], fontsize=fsz)
            ax.set_ylabel(labels[1], fontsize=fsz)

        ax.tick_params(
            axis='both', which='both', bottom=False, top=False, left=False,
            right=False
        )
        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_major_formatter('{x:.1f}')
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.yaxis.set_major_formatter('{x:.1f}')
