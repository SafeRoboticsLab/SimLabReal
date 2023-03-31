# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Advanced-Dense environment.

This environment is used for Advanced-Dense setting, where we load fixed room layout (square) and randomly placed furniture from the 3D-FRONT dataset and a simulated robot navigates. The robot action include forward velocity and yaw velocity.

Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
         Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import numpy as np
import random

from env.advanced_env import AdvancedEnv


class AdvancedDenseEnv(AdvancedEnv):

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
        mesh_parent_path='/home/temp/data/3D-FUTURE-model',
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
        # Specific to dense env
        task_parent_path='/home/temp/data/3d-front-tasks-advanced-dense',
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
            task_parent_path (str): Path to the task parent directory.
        """
        self.task_parent_path = task_parent_path
        super(AdvancedDenseEnv, self).__init__(
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
            use_simplified_mesh=use_simplified_mesh,
            mesh_parent_path=mesh_parent_path,
            fill_obs=fill_obs,
            camera_tilt=camera_tilt,
            camera_tilt_range=camera_tilt_range,
            camera_tilt_noise_std=camera_tilt_noise_std,
            camera_roll_noise_std=camera_roll_noise_std,
            camera_yaw_noise_std=camera_yaw_noise_std,
            perf_xdot_range=perf_xdot_range,
            backup_xdot_range=backup_xdot_range,
            max_x_acc=max_x_acc,
            max_theta_acc=max_theta_acc,
        )

    def set_default_task(self, task):
        """
        Set default task.
        
        Args:
            task (dict): Task dictionary.
        """
        self._task = task
        self._goal_loc = task.get('goal_loc', np.array([0., 0.]))
        self._goal_radius = task.get('goal_radius', 0.50)
        self._init_state = task.get('init_state', np.array([0.0, 0.0, 0.0]))
        self._init_goal_dist = np.linalg.norm(
            self._init_state[:2] - self._goal_loc
        )
        self._task_id = task.get('task_id', 0)
        self._piece_id_all = task.get('piece_id_all', [])
        self._piece_pos_all = task.get('piece_pos_all', [])
        self._task_path = os.path.join(
            self.task_parent_path, str(self._task_id)
        )
        self._mesh_bounds = task.get(
            'mesh_bounds', np.array([[-2, 2], [-2, 2]])
        )
        self.bounds = np.append(self._mesh_bounds, [[0, 2 * np.pi]], axis=0)
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

    def reset_obstacles(self, task):
        """
        Overriding - load square room layout by generating rectangular shapes in PyBullet
        
        Args:
            task (dict): Task dictionary.
        """
        if 'task_id' not in task:
            return
        task_path = os.path.join(self.task_parent_path, str(task['task_id']))

        # Skip if same task already loaded
        if len(self._obs_id_all) > 0 and task_path == self._task_path:
            return

        self._task_path = task_path
        self._task_id = task['task_id']
        self._mesh_bounds = task['mesh_bounds']
        self._piece_id_all = task.get('piece_id_all', [])
        self._piece_pos_all = task.get('piece_pos_all', [])
        self.bounds = np.append(self._mesh_bounds, [[0, 2 * np.pi]], axis=0)
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Remove existing ones
        for obs_id in self._obs_id_all:
            self._p.removeBody(obs_id)
        self._obs_id_all = []

        # Load wall and ceiling
        room_obj_all = [
            os.path.join(self._task_path, 'floor.obj'),
            os.path.join(self._task_path, 'wall.obj'),
        ]
        for obj_path in room_obj_all:
            obj_collision_id = self._p.createCollisionShape(
                self._p.GEOM_MESH,
                fileName=obj_path,
                flags=self._p.GEOM_FORCE_CONCAVE_TRIMESH  # concave
            )
            obj_visual_id = self._p.createVisualShape(
                self._p.GEOM_MESH, fileName=obj_path
            )
            obj_id = self._p.createMultiBody(
                baseMass=0,  # static, allow concave
                baseCollisionShapeIndex=obj_collision_id,
                baseVisualShapeIndex=obj_visual_id,
                baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0])
            )
            self._obs_id_all += [obj_id]

        # Load furniture
        for piece_id, piece_pos in zip(
            self._piece_id_all, self._piece_pos_all
        ):
            piece_path = os.path.join(
                self.mesh_parent_path, piece_id, self.piece_mesh_name
            )
            obj_collision_id = self._p.createCollisionShape(
                self._p.GEOM_MESH,
                fileName=piece_path,
                flags=self._p.GEOM_FORCE_CONCAVE_TRIMESH  # concave
            )
            obj_visual_id = self._p.createVisualShape(
                self._p.GEOM_MESH, fileName=piece_path
            )
            obj_id = self._p.createMultiBody(
                baseMass=0,  # static, allow concave
                baseCollisionShapeIndex=obj_collision_id,
                baseVisualShapeIndex=obj_visual_id,
                basePosition=[piece_pos[0], piece_pos[1], 0],
                baseOrientation=self._p.getQuaternionFromEuler([
                    np.pi / 2, 0, 0
                ])
            )
            self._obs_id_all += [obj_id]

        # Add white patches to walls
        add_wall = [random.randint(0, 1) for _ in range(4)]
        color = np.random.uniform(0.8, 1)
        if add_wall[0]:  # back
            width = np.random.uniform(1, 4)
            height = np.random.uniform(2, 3)
            y_loc = np.random.uniform(-3, 3)
            obj_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX,
                halfExtents=[0.005, width / 2, height / 2],
                rgbaColor=[color, color, color, 1],
                specularColor=[1, 1, 1],
            )  # white patch
            obj_id = self._p.createMultiBody(
                baseMass=0,  # static
                baseVisualShapeIndex=obj_visual_id,
                basePosition=[0.005, y_loc, height / 2],
                # basePosition=[2, 3.45, 0.5],
                baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0])
            )
            self._obs_id_all += [obj_id]
        if add_wall[1]:  # left
            width = np.random.uniform(1, 4)
            height = np.random.uniform(2, 3)
            x_loc = np.random.uniform(1, 6)
            obj_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX,
                halfExtents=[0.005, width / 2, height / 2],
                rgbaColor=[color, color, color, 1],
                specularColor=[1, 1, 1],
            )  # white patch
            obj_id = self._p.createMultiBody(
                baseMass=0,  # static
                baseVisualShapeIndex=obj_visual_id,
                basePosition=[x_loc, 3.495, height / 2],
                baseOrientation=self._p.getQuaternionFromEuler([
                    0, 0, np.pi / 2
                ])
            )
            self._obs_id_all += [obj_id]
        if add_wall[2]:  # right
            width = np.random.uniform(1, 4)
            height = np.random.uniform(2, 3)
            x_loc = np.random.uniform(1, 6)
            obj_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX,
                halfExtents=[0.005, width / 2, height / 2],
                rgbaColor=[color, color, color, 1],
                specularColor=[1, 1, 1],
            )  # white patch
            obj_id = self._p.createMultiBody(
                baseMass=0,  # static
                baseVisualShapeIndex=obj_visual_id,
                basePosition=[x_loc, -3.495, height / 2],
                baseOrientation=self._p.getQuaternionFromEuler([
                    0, 0, np.pi / 2
                ])
            )
            self._obs_id_all += [obj_id]
        if add_wall[3]:  # back
            width = np.random.uniform(1, 4)
            height = np.random.uniform(2, 3)
            y_loc = np.random.uniform(-3, 3)
            obj_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX,
                halfExtents=[0.005, width / 2, height / 2],
                rgbaColor=[color, color, color, 1],
                specularColor=[1, 1, 1],
            )  # white patch
            obj_id = self._p.createMultiBody(
                baseMass=0,  # static
                baseVisualShapeIndex=obj_visual_id,
                basePosition=[6.995, y_loc, height / 2],
                baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0])
            )
            self._obs_id_all += [obj_id]

        if self.render:
            if "_goal_id" in vars(self).keys():
                self._p.removeBody(self._goal_id)
            goal_visual_id = self._p.createVisualShape(
                self._p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
                rgbaColor=[0, 1, 0, 1]
            )
            self._goal_id = self._p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=goal_visual_id,
                basePosition=[self._goal_loc[0], self._goal_loc[1], 0.05],
                baseOrientation=self._p.getQuaternionFromEuler([0, 0, 0])
            )
