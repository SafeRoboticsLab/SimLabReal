# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Advanced environment.

This environment is used for Advanced-Realistic setting, where we load professionally-designed indoor home layout from the 3D-FRONT dataset and a simulated robot navigates. The robot action include forward velocity and yaw velocity.

This class serves the parent class for AdvancedDenseEnv for Advanced-Dense setting.

Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
         Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import random

import pybullet as pb
from pybullet_utils import bullet_client as bc

from env.base_env import BaseEnv
from utils.misc import scale_and_shift, plot_circle


class AdvancedEnv(BaseEnv):

    def __init__(
        self,
        task=None,
        render=True,
        use_rgb=True,
        img_h=128,
        img_w=128,
        use_append=True,
        obs_buffer=0.,
        boundary_buffer=0.,
        g_x_fail=1,
        num_traj_per_visual_init=1,
        fixed_init=False,
        sparse_reward=False,
        sample_inside_obs=False,
        max_step_train=100,
        max_step_eval=100,
        done_type='fail',
        terminal_type='const',
        reward_type='all',
        reward_goal=10.,
        reward_obs=-5.,
        reward_wander=2.,
        # Specific to advanced envs
        use_simplified_mesh=False,
        mesh_parent_path='/home/temp/data/3d-front-house-meshes/',
        fill_obs=True,
        camera_tilt=15,
        camera_tilt_range=None,
        camera_tilt_noise_std=0,
        camera_roll_noise_std=0,
        camera_yaw_noise_std=0,
        perf_xdot_range=[0.5, 1.0],
        backup_xdot_range=[0.2, 0.5],
        max_x_acc=0,
        max_theta_acc=0,
        **kwargs,
    ):
        """
        Args:
            task (dict): Task dictionary.
            render (bool): Whether to render the environment.
            use_rgb (bool): Whether to use RGB image as observation.
            img_h (int): Image height.
            img_w (int): Image width.
            use_append (bool): Whether to append the robot state to the observation.
            obs_buffer (float): Buffer around the obstacle.
            boundary_buffer (float): Buffer around the boundary.
            g_x_fail (float): Threshold for failure.
            num_traj_per_visual_init (int): Number of trajectories per visual initialization.
            fixed_init (bool): Whether to use fixed initialization.
            sparse_reward (bool): Whether to use sparse reward.
            sample_inside_obs (bool): Whether to sample inside the obstacle.
            max_step_train (int): Maximum number of steps for training.
            max_step_eval (int): Maximum number of steps for evaluation.
            done_type (str): Type of done.
            terminal_type (str): Type of terminal.
            reward_type (str): Type of reward.
            reward_goal (float): Reward for reaching the goal.
            reward_obs (float): Reward for collision.
            reward_wander (float): Reward for wandering.
            use_simplified_mesh (bool): Whether to use simplified mesh.
            mesh_parent_path (str): Path to the mesh parent directory.
            fill_obs (bool): Whether to fill the obstacle.
            camera_tilt (float): Camera tilt angle.
            camera_tilt_range (list): Camera tilt angle range.
            camera_tilt_noise_std (float): Camera tilt angle noise standard deviation.
            camera_roll_noise_std (float): Camera roll angle noise standard deviation.
            camera_yaw_noise_std (float): Camera yaw angle noise standard deviation.
            perf_xdot_range (list): Performance xdot range.
            backup_xdot_range (list): Backup xdot range.
            max_x_acc (float): Maximum x acceleration.
            max_theta_acc (float): Maximum theta acceleration.
        """
        super(AdvancedEnv, self).__init__(
            task=task,
            render=render,
            use_rgb=use_rgb,
            img_h=img_h,
            img_w=img_w,
            use_append=use_append,
            obs_buffer=obs_buffer,
            boundary_buffer=boundary_buffer,
            g_x_fail=g_x_fail,
            num_traj_per_visual_init=num_traj_per_visual_init,
            fixed_init=fixed_init,
            sparse_reward=sparse_reward,
            sample_inside_obs=sample_inside_obs,
            max_step_train=max_step_train,
            max_step_eval=max_step_eval,
            done_type=done_type,
            terminal_type=terminal_type,
            reward_type=reward_type,
            reward_goal=reward_goal,
            reward_obs=reward_obs,
            reward_wander=reward_wander,
        )
        # Specific to room envs
        self.mesh_parent_path = mesh_parent_path
        self.fill_obs = fill_obs
        if use_simplified_mesh:
            self.piece_mesh_name = 'raw_model_simplified.obj'
        else:
            self.piece_mesh_name = 'raw_model.obj'
        self.perf_xdot_range = perf_xdot_range
        self.backup_xdot_range = backup_xdot_range
        self.max_x_acc = max_x_acc
        self.max_theta_acc = max_theta_acc

        # Define continuous space limit
        self.action_lim = np.float32(np.array([1.]))

        # Define robot dimensions and dynamics
        self.robot_dim = [
            0.21, 0.15, 0.18
        ]  # spirit body width is 0.12 in half - motors extruded
        self.robot_com_height = self.robot_dim[2]
        self.dt = 0.1  # 10 Hz for now

        # Define observation info
        self.camera_x_offset = self.robot_dim[0] + 0.01
        self.camera_height = 0.45  # 13.5 inch / 34cm from floor as measured on the real robot with normal gait stance - but a bit higher now with new camera
        self.camera_aspect = 16 / 9  # 3/4
        # self.camera_fov = 42  # for D435i on spirit, using 640x480 (4:3), HFOV is 55 deg from https://github.com/IntelRealSense/librealsense/issues/7170. OpenGL uses VFOV, which is now 2*arctan(3/4*tan(55 deg/2))=42.65 deg
        self.camera_fov = 72  # HD720 on ZED 2; 72 VFOV and 104 HFOV

        # Overriding base
        self.camera_tilt = camera_tilt
        self.camera_tilt_range = camera_tilt_range
        self.camera_tilt_noise_std = camera_tilt_noise_std
        self.camera_roll_noise_std = camera_roll_noise_std
        self.camera_yaw_noise_std = camera_yaw_noise_std

        # Extract task info - overwritten by reset
        if task is None:
            task = {}
        self.set_default_task(task)
        self._obs_id_all = []

    @property
    def env_type(self):
        return 'advanced'

    @property
    def state_dim(self):
        return 3

    @property
    def action_dim(self):
        return 2

    def _get_obs(self, state):
        """
        Fill the image of white color (mesh holes in the walls ) with random color

        Args:
            state (np.ndarray): State of the environment.
        
        Returns:
            obs (np.ndarray): Observation of the environment.
        """
        obs = super()._get_obs(state)
        if self.fill_obs:
            color = random.randint(120, 220)
            obs_sum = np.sum(obs, axis=0) >= 765
            obs = np.where(obs_sum, color, obs)
        return obs

    def report(self, **kwargs):
        pass

    def init_pb(self):
        """
        Initialize the PyBullet client.
        """
        if self.render:
            self._p = bc.BulletClient(connection_mode=pb.GUI)
        else:
            self._p = bc.BulletClient()
        self._physics_client_id = self._p._client
        p = self._p
        p.resetSimulation()
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.8)
        if self.render:
            p.resetDebugVisualizerCamera(3.0, 180, -89, [0, 0, 0])

    def close_pb(self):
        """
        Close the PyBullet client.
        """
        if "_goal_id" in vars(self).keys():
            self._p.removeBody(self._goal_id)
            del self._goal_id
        if "robot_id" in vars(self).keys():
            self._p.removeBody(self.robot_id)
            del self.robot_id
        for obs_id in self._obs_id_all:
            self._p.removeBody(obs_id)
        self._obs_id_all = []
        self._p.disconnect()
        self._physics_client_id = -1

    def set_default_task(self, task):
        """
        Set the default task.
        
        Args:
            task (dict): Task to be set.
        """
        self._task = task
        self._goal_loc = task.get('goal_loc', np.array([0., 0.]))
        self._goal_radius = task.get('goal_radius', 0.50)
        self._init_state = task.get('init_state', np.array([0.0, 0.0, 0.0]))
        self._init_goal_dist = np.linalg.norm(
            self._init_state[:2] - self._goal_loc
        )
        self._mesh_path = task.get(
            'mesh_path',
            os.path.join(
                '/home', 'temp', 'data', '3d-front', '3d-front-obj',
                'e9c4facf-557a-4002-8d8f-acce694871c2'
            )
        )
        self._room_names = task.get('room_names', [0, 1])
        self._mesh_bounds = task.get(
            'mesh_bounds', np.array([[-2, 2], [-2, 2]])
        )
        self.bounds = np.append(self._mesh_bounds, [[0, 2 * np.pi]], axis=0)
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

    def reset_goal(self):
        """
        Unlike Vanilla environments, here we do not reset goals since the goal is specified for each task/layout.
        """
        raise NotImplementedError

    def reset_task(self, task):
        """
        Reset the task.
        
        Args:
            task (dict): Task to be reset.
        """
        self._task = task
        if 'goal_loc' in task:
            self._goal_loc = task['goal_loc']
        if 'goal_radius' in task:
            self._goal_radius = task['goal_radius']
        if 'init_state' in task:
            self._init_state = task['init_state']
        self._init_goal_dist = np.linalg.norm(
            self._init_state[:2] - self._goal_loc
        )

        # Reset mesh
        self.reset_obstacles(task)

    def reset_obstacles(self, task):
        """
        Reset the obstacles.
        
        Args:
            task (dict): Task to be reset.
        """
        if 'mesh_path' not in task:
            return
        task_mesh_path = os.path.join(
            self.mesh_parent_path, task['mesh_path'].split('/')[-1]
        )
        if len(self._obs_id_all) > 0 and task_mesh_path == self._mesh_path:
            return  # same mesh already loaded
        self._mesh_path = task_mesh_path
        self._room_names = task['room_names']
        self._mesh_bounds = task['mesh_bounds']
        self.bounds = np.append(self._mesh_bounds, [[0, 2 * np.pi]], axis=0)
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Remove existing ones
        for obs_id in self._obs_id_all:
            self._p.removeBody(obs_id)
        self._obs_id_all = []

        # x/y only, excludes objs that are too high to collide with robot
        # (e.g. ceiling) and too low (e.g. floor)
        for room_name in self._room_names:
            room_obj_all = glob.glob(
                os.path.join(self._mesh_path, room_name, '*.obj')
            )
            for obj_path in room_obj_all:
                obj_collision_id = self._p.createCollisionShape(
                    self._p.GEOM_MESH,
                    fileName=obj_path,
                    flags=self._p.GEOM_FORCE_CONCAVE_TRIMESH,  # concave
                )
                obj_visual_id = self._p.createVisualShape(
                    self._p.GEOM_MESH, fileName=obj_path
                )
                obj_id = self._p.createMultiBody(
                    baseMass=0,  # static
                    baseCollisionShapeIndex=obj_collision_id,
                    baseVisualShapeIndex=obj_visual_id,
                    baseOrientation=self._p.getQuaternionFromEuler([
                        np.pi / 2, 0, 0
                    ])
                )
                self._obs_id_all += [obj_id]

        if self.render:
            if "_goal_id" in vars(self).keys():
                self._p.removeBody(self._goal_id)
            goal_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05],
                rgbaColor=[0, 1, 0, 1],
            )
            self._goal_id = self._p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=goal_visual_id,
                basePosition=[self._goal_loc[0], self._goal_loc[1], 0.05],
                baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0]),
            )

    def reset_robot(self, state):
        """
        Reset the robot.
        
        Args:
            state (np.ndarray): State to be reset.
        """
        if "robot_id" in vars(self).keys():
            self._p.resetBasePositionAndOrientation(
                self.robot_id,
                posObj=np.append(state[:2], self.robot_com_height),
                ornObj=self._p.getQuaternionFromEuler([0, 0, state[2]])
            )
        else:
            robot_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX, halfExtents=self.robot_dim,
                rgbaColor=[0, 0, 1, 1]
            )
            self.robot_id = self._p.createMultiBody(
                baseMass=0,  # static
                baseVisualShapeIndex=robot_visual_id,
                basePosition=np.append(state[:2], self.robot_com_height),
                baseOrientation=self._p.getQuaternionFromEuler([
                    0, 0, state[2]
                ])
            )
        self.prev_x_dot = 0
        self.prev_theta_dot = 0

    def reset(self, random_init=False, state_init=None, task=None):
        """
        Reset the environment.
        
        Args:
            random_init (bool): Whether to use random initial state.
            state_init (np.ndarray): Initial state.
            task (dict): Task to be reset.
        """
        # Start PyBullet session if first time - and set obstacle
        if self._physics_client_id < 0:
            self.init_pb()

        # bit hacky since in room env, init_state is part of the task
        if task is None:
            task = self._task
        else:  # we at least reset to default_task in init
            self._task = task
        self.reset_task(task)

        # a bit weird - but basically always using init_state
        if state_init is not None:
            self._state = state_init
        elif random_init:
            self._state = self.sample_state(sample_random=True)
        elif self.fixed_init:
            self._init_state = task.get(
                'init_state', np.array([0.0, 0.0, 0.0])
            )
        else:
            raise ValueError("Need one method to get an initial state")

        # Update init
        self._state = self._init_state.copy()

        # Reset timer
        self.step_elapsed = 0

        # Reset robot
        self.reset_robot(self._state.copy())

        return self._get_obs(self._state)

    def get_control(self, action):
        """
        Convert raw policy output [-1,1] to actual robot control specified by the ranges.
        
        Args:
            action (np.ndarray): Raw action.
        
        Returns:
            np.ndarray: Control.
        """
        theta_dot = action[1]

        # Map xdot to some positive range
        if action[-1] == 1:
            x_dot_range = self.perf_xdot_range
        else:
            x_dot_range = self.backup_xdot_range
        x_dot = scale_and_shift(action[0], [-1, 1], x_dot_range)

        # No translation for now
        y_dot = 0

        # Map thetadot to both positive and negative range, but clip [-theta_dot_thres, theta_dot_thres] to zero
        # if abs(theta_dot) < theta_dot_thres:
        #     theta_dot = 0
        # else:
        #     theta_dot = (theta_dot - sign(theta_dot) * theta_dot_thres) / (1-theta_dot_thres) * 0.5 + sign(theta_dot) * 0.5

        # model acceleration
        if self.max_x_acc > 0:
            if x_dot > self.prev_x_dot:
                x_dot = min(self.prev_x_dot + self.max_x_acc, x_dot)
            else:
                x_dot = max(self.prev_x_dot - self.max_x_acc, x_dot)
        if self.max_theta_acc > 0:
            if theta_dot > self.prev_theta_dot:
                theta_dot = min(
                    self.prev_theta_dot + self.max_theta_acc, theta_dot
                )
            else:
                theta_dot = max(
                    self.prev_theta_dot - self.max_theta_acc, theta_dot
                )

        return [x_dot, y_dot, theta_dot]

    def move_robot(self, action, state, return_converted=False):
        """
        Move the robot with simple Dubins dynamics. Right-hand coordinates.

        Args:
            action (np.ndarray): to be applied.

        Returns:
            state: after action applied
        """
        # Integrate
        x, y, theta = state
        x_dot, y_dot, theta_dot = self.get_control(action)
        x_new = (
            x + x_dot * np.cos(theta) * self.dt
            - y_dot * np.sin(theta) * self.dt
        )
        y_new = (
            y + x_dot * np.sin(theta) * self.dt
            + y_dot * np.cos(theta) * self.dt
        )
        theta_new = theta + theta_dot * self.dt
        if theta_new > 2 * np.pi:
            theta_new -= 2 * np.pi
        elif theta_new < 0:
            theta_new += 2 * np.pi
        state = np.array([x_new, y_new, theta_new])

        # Move robot
        self._p.resetBasePositionAndOrientation(
            self.robot_id, posObj=np.append(state[:2], self.robot_com_height),
            ornObj=self._p.getQuaternionFromEuler([0, 0, state[2]])
        )
        self._p.stepSimulation()

        # record prev action
        self.prev_x_dot = x_dot
        self.prev_theta_dot = theta_dot

        if return_converted:
            return state, [x_dot, y_dot, theta_dot]
        else:
            return state

    def sample_state(self, sample_random=False, **kwargs):
        """
        Sample a random state.
        
        Args:
            sample_random (bool): Whether to sample randomly.
        
        Returns:
            np.ndarray: Sampled state.
        """
        if sample_random:
            theta_rnd = 2.0 * np.pi * self.rng.uniform()
            flag = True
            low = self.low[:2]
            high = self.high[:2]
            cnt_attempt = 0
            while flag:
                rnd_state = self.rng.uniform(low=low, high=high)
                l_x = self.target_margin(rnd_state)
                g_x = self.safety_margin(np.append(rnd_state, theta_rnd))
                # k_x = np.linalg.norm(rnd_state - self._init_state[:2])
                # possible outside rooms right now

                if g_x > -0.3 or l_x < 0.5:
                    # if g_x > -0.3 or l_x < 0.5 or k_x < 1:
                    flag = True
                else:
                    flag = False

                if cnt_attempt > 1000:
                    return self._init_state
                cnt_attempt += 1
            x_rnd, y_rnd = rnd_state
            return np.array([x_rnd, y_rnd, theta_rnd])
        else:
            return self._init_state

    def step(self, action):
        """
        Step the environment.
        
        Args:
            action (np.ndarray): Action to be applied.
        
        Returns:
            np.ndarray: Observation.
            float: Reward.
            bool: Whether the episode is done.
            dict: Additional information.
        """
        # Move car in simulation
        self._state, _ = self.move_robot(
            action, self._state, return_converted=True
        )

        # = `l_x` and `g_x` signal
        info = self._get_info(self._state)
        l_x = info['l_x']
        g_x = info['g_x']
        heading = info['heading']
        fail = (
            g_x > 0
        )  # prevent bad image at the boundary - small value to buffer
        success = l_x <= 0

        dist_to_goal_center = np.linalg.norm(self._state[:2] - self._goal_loc)
        # = Sparse `reward` - small penalty for wandering around
        if self.sparse_reward:
            reward = self.reward_wander

            # Large reward for reaching target
            if dist_to_goal_center < self._goal_radius:
                reward = self.reward_goal
        else:
            # = Dense `reward`
            if dist_to_goal_center < self._goal_radius:

                # Normalized - need k_x technically, but in practice works well
                # without k_x, like adding noise to reward
                reward = self.reward_goal * self._init_goal_dist

                # Unnormalized
                # reward = self.reward_goal
            else:

                # Normalized - need k_x technically
                # reward = (
                #     self.reward_wander -
                #     (self._init_goal_dist - dist_to_goal_center)
                #     / self._init_goal_dist
                # )

                # Unnormalized
                reward = self.reward_wander * (5-dist_to_goal_center)

            # Add action penalty
            # x_dot = action_converted[0]  # 0-1
            # theta_dot = action_converted[1]  # [-1,-0.5] and [0.5,1]
            # if x_dot < 0.5:
            #     reward -= (1 - x_dot)
            # if abs(theta_dot) > 0.8:
            #     reward -= abs(theta_dot)

        # Obstacle penalty
        if self.reward_type == 'all' and fail:
            reward = self.reward_obs

        # = `info` signal
        # for recovery RL
        binary_cost = 0
        if fail:
            binary_cost = 1

        # = `done` signal
        if self.done_type == 'end':
            done = not self.check_within_bounds(self._state)
        elif self.done_type == 'fail':
            done = fail
        elif self.done_type == 'TF':
            done = fail or success
        else:
            raise ValueError("invalid done type")

        # Count time
        self.step_elapsed += 1
        if self.step_elapsed == self.max_steps:
            done = True

        if self.done_type == 'fail' and fail:
            g_x = self.g_x_fail

        # Return
        info_out = {
            # 'task': self._task,
            'state': self._state,
            'g_x': g_x,
            'l_x': l_x,
            'k_x': np.linalg.norm(self._state[:2] - self._init_state[:2]),
            'heading': heading,
            'binary_cost': binary_cost,
        }

        return self._get_obs(self._state), reward, done, info_out

    def safety_margin(self, state, return_boundary=False):
        """
        Compute the margin (e.g. distance) between state and failue set.

        Args:
            state: consisting of [x, y, theta].

        Returns:
            g(state): safety margin, positive value suggests inside of the
                failure set.
        """
        state_xyz = np.array([state[0], state[1], self.robot_com_height])

        # Do ray check to sky to see if out of bound
        rayOutput = self._p.rayTest(
            state_xyz,
            state_xyz
            + np.array([0, 0, 10.0])  # no floor should be higher than 10m...
        )
        if rayOutput[0][0] < 0:
            boundary_margin = 0  # out of bound basically
        else:
            boundary_margin = max(
                self.bounds[0, 0] - state[0], state[0] - self.bounds[0, 1],
                self.bounds[1, 0] - state[1], state[1] - self.bounds[1, 1]
            )
        # rot_matrix = [0, 0, state[2]]
        # rot_matrix = self._p.getMatrixFromQuaternion(
        #     self._p.getQuaternionFromEuler(rot_matrix)
        # )
        # rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # rayOutput = self._p.rayTest(
        #     state_xyz, state_xyz + rot_matrix.dot(np.array([1, 0, 1]))
        # )
        # if rayOutput[0][0] > -1:
        #     print(rayOutput[0][2])
        # else:
        #     print('inf')

        # g_xList = [boundary_margin]
        # for obs_id, obs_box in self._obj_id_loc_for_check.items():
        #     # for x in self._p.getClosestPoints(
        #     #     self.robot_id, obs_id, self.col_threshold
        #     # ):
        #     #     print(x[6], x[8])
        #     # contactDistance, positive for separation, negative for
        #     # penetration; # exclude if contactPoint is below z=0.01 on the
        #     # object - sometimes the wall/room mesh has some residue on the
        #     # floor
        #     dist = [
        #         x[8]
        #         for x in self._p.
        #         getClosestPoints(self.robot_id, obs_id, self.col_threshold)
        #         if x[6][2] > 0.01
        #     ]
        #     if len(dist) > 0:
        #         print(obs_id, obs_box)
        # g_xList += dist
        # print(g_xList)
        numRays = 360  # should be plenty - can try fewer
        rayLen = 10  # the max dimension of the room mesh is 10m right now
        rayTo = [
            [
                rayLen * math.sin(2. * math.pi * float(i) / numRays)
                + state[0],
                rayLen * math.cos(2. * math.pi * float(i) / numRays)
                + state[1],
                self.robot_com_height  # parallel
            ] for i in range(numRays)
        ]

        # Add rays at angles
        yaw_all = np.arange(-np.pi, np.pi + 0.01, np.pi / 8)
        # yaw_all = [
        #     state[2] - np.pi / 2, state[2] - np.pi / 4, state[2],
        #     state[2] + np.pi / 4, state[2] + np.pi / 2
        # ]
        angles = np.arange(0, np.pi / 2 + 0.01, np.pi / 16)
        for yaw in yaw_all:
            for angle in angles:
                rot_matrix = self._p.getMatrixFromQuaternion(
                    self._p.getQuaternionFromEuler([0, -angle, yaw])
                )
                rot_matrix = np.array(rot_matrix).reshape(3, 3)
                vector = rot_matrix.dot([1, 0, 0])
                rayTo += [state_xyz + vector / np.linalg.norm(vector) * rayLen]
        numRaysAngled = len(angles) * len(yaw_all)
        numRays += numRaysAngled

        # Get rays
        rayFrom = [state_xyz] * numRays
        rayOutput = self._p.rayTestBatch(
            rayFrom,
            rayTo,
        )  # 1st hit for each ray
        min_hit_dist = min([out[2] for out in rayOutput]) * rayLen
        argmin_hit_dist = np.argmin([out[2] for out in rayOutput])
        # Consider collision if there is an obstacle 2m above
        # above_dist = rayOutput[-1][2] * rayLen
        # if above_dist < 2:
        #     min_hit_dist = 0

        # Debug
        # out_num = 0
        # for ind in range(numRays):
        #     self._p.addUserDebugLine(rayFrom[ind], rayTo[ind])
        # for ind, out in enumerate(rayOutput):
        #     if out[0] != -1:
        #         out_num += 1
        #         print(out[2] * rayLen)
        #         point = (
        #             state_xyz + rayLen * out[2] *
        #             (np.array(rayTo[ind]) - np.array(state_xyz)) /
        #             np.linalg.norm(np.array(rayTo[ind]) - np.array(state_xyz))
        #         )
        #         if ind != argmin_hit_dist:
        #             color = [1, 0, 0]
        #         else:
        #             color = [0, 1, 0]
        #         self._p.addUserDebugLine(
        #             state_xyz, point, lineColorRGB=color, lineWidth=10
        #         )
        # while 1:
        #     continue

        obstacle_margin = np.max(-min_hit_dist)
        safety_margin = max(obstacle_margin, boundary_margin)

        # Smooth
        safety_margin += self.obs_buffer

        if return_boundary:
            return safety_margin, boundary_margin, obstacle_margin
        else:
            return safety_margin

    # == Plotting ==
    def visualize(
        self, q_func, policy, mode, rndTraj=False, num_rnd_traj=10,
        end_type='TF', vmin=-1, vmax=1, nx=51, ny=51, cmap='seismic',
        labels=None, bool_plot=False, plot_v=True, plot_contour=False,
        normalize_v=False, traj_size=0, frame_skip=0, latent_dist=None,
        task=None, revert_task=True, shield=False, backup=None,
        shield_dict=None
    ):
        """
        Visualize trajectories and value function.

        Args:
            q_func (object): agent's Q-network.
            policy (func): agent's policy.
            rndTraj (bool, optional): random initialization or not.
                Defaults to False.
            num_rnd_traj (int, optional): number of states. Defaults to None.
            vmin (int, optional): vmin in colormap. Defaults to -1.
            vmax (int, optional): vmax in colormap. Defaults to 1.
            nx (int, optional): # points in x-axis. Defaults to 101.
            ny (int, optional): # points in y-axis. Defaults to 101.
            cmap (str, optional): color map. Defaults to 'seismic'.
            labels (list, optional): x- and y- labels. Defaults to None.
            bool_plot (bool, optional): plot the binary values.
                Defaults to False.
        """
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()

        # Set task in environments; get states
        if revert_task:
            prev_task = self._task
        if task is not None:
            self.reset_task(task)
        theta = self._init_state[-1]
        visual_initial_states = np.tile(
            self._init_state, (self.num_traj_per_visual_init, 1)
        )

        # == Plot saved occupancy ==
        voxel_grid = self._task['voxel_grid']
        grid_pitch = self._task['grid_pitch']
        x = np.arange(
            self._mesh_bounds[0, 0], self._mesh_bounds[0, 1], grid_pitch
        )  # bounds have row for a dimension
        y = np.arange(
            self._mesh_bounds[1, 0], self._mesh_bounds[1, 1], grid_pitch
        )
        if voxel_grid.shape[0] > len(x):  # hack
            num_diff = voxel_grid.shape[0] - len(x)
            for _ in range(num_diff):
                x = np.append(x, x[-1] + grid_pitch)
        elif voxel_grid.shape[0] < len(x):
            num_diff = len(x) - voxel_grid.shape[0]
            x = x[:-num_diff]
        if voxel_grid.shape[1] > len(y):
            num_diff = voxel_grid.shape[1] - len(y)
            for _ in range(num_diff):
                y = np.append(y, y[-1] + grid_pitch)
        elif voxel_grid.shape[1] < len(y):
            num_diff = len(y) - voxel_grid.shape[1]
            y = y[:-num_diff]
        ax.pcolormesh(
            x, y, voxel_grid.T, shading='nearest', cmap='Greys', alpha=0.3
        )

        # == Plot failure / target set ==
        plot_circle(
            self._goal_loc, self._goal_radius, ax, c='m', lw=3, zorder=0
        )

        # For get_value()
        latent = None
        if latent_dist is not None:
            latent = latent_dist.mean.unsqueeze(0)

        # Check mixed stack
        if policy.obs_channel == 3:
            traj_size = 0

        # Use more skip if backup
        if mode == 'safety':
            frame_skip = int(frame_skip * 1.5)

        # == Plot V ==
        if plot_v:
            self.plot_v_values(
                q_func, fig, ax, theta=theta, bool_plot=bool_plot,
                plot_cbar=True, vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
                normalize_v=normalize_v, plot_contour=plot_contour, alpha=0.3,
                latent=latent, traj_size=traj_size
            )

        # == Plot Trajectories ==
        sim_traj_args = dict(
            end_type=end_type,
            traj_size=traj_size,
            frame_skip=frame_skip,
            latent_dist=latent_dist,
            shield=shield,
            backup=backup,
            shield_dict=shield_dict,
        )

        if rndTraj:
            # do not specify theta - theta only used when sampling states in
            # toy env
            sample_args = dict()
            if mode == 'safety' or mode == 'safety_ra' or mode == 'risk':
                sample_args['sample_random'] = True
            self.plot_trajectories(
                policy, mode, ax, num_rnd_traj=num_rnd_traj,
                sample_args=sample_args, sim_traj_args=sim_traj_args,
                plot_dot=True, print_step=True
            )
        else:
            self.plot_trajectories(
                policy, mode, ax, states=visual_initial_states,
                sim_traj_args=sim_traj_args, plot_dot=True, print_step=True
            )

        # == Formatting ==
        self.plot_formatting(ax, labels=labels, use_equal_aspect_ratio=False)
        # fig.tight_layout()

        # Task
        if revert_task:
            self.reset_task(prev_task)

        return fig
