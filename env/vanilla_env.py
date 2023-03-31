# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client as bc
import matplotlib.pyplot as plt

from env.base_env import BaseEnv
from utils.misc import (
    plot_circle, plot_rect, index_to_state, state_to_index, scale_and_shift
)


class VanillaEnv(BaseEnv):
    """
    Simple 2D navigation with circular obstacle using PyBullet. No dynamics or
    collision simulation.
    """

    def __init__(
        self,
        task=None,
        render=False,
        use_rgb=True,
        img_h=128,
        img_w=128,
        use_append=False,
        obs_buffer=0.,
        boundary_buffer=0.,
        g_x_fail=1.0,
        num_traj_per_visual_init=1,
        fixed_init=False,
        sparse_reward=False,
        sample_inside_obs=False,
        sample_inside_tar=True,
        uniform_wall_color=False,
        max_step_train=100,
        max_step_eval=100,
        done_type='fail',
        terminal_type='const',
        reward_type='all',
        reward_goal=10.,
        reward_obs=-5.,
        reward_wander=2.,
        has_ceiling=True,
        **kwargs,
    ):
        """
        Args:
            task (str, optional): the name of the task. Defaults to None.
            img_h (int, optional): the height of the image. Defaults to 128.
            img_w (int, optional): the width of the image. Defaults to 128.
            obs_buffer (float, optional): the buffer distance between the
                robot and the obstacle. Defaults to 0.0.
            boundary_buffer (float, optional): the buffer distance between the
                robot and the boundary. Defaults to 0.0.
            num_traj_per_visual_init (int, optional): the number of
                trajectories to be visualized in the initial state.
                Defaults to 1.
            fixed_init (bool, optional): whether to use the fixed initial
                state. Defaults to False.
            sparse_reward (bool, optional): whether to use the sparse reward.
                Defaults to False.
            use_rgb (bool, optional): whether to use RGB image. Defaults to
                True.
            render (bool, optional): whether to render the environment.
                Defaults to True.
            sample_inside_obs (bool, optional): whether to sample the initial
                state inside the obstacle. Defaults to False.
            max_step_train (int, optional): the maximum number of steps to
                train. Defaults to 100.
            max_step_eval (int, optional): the maximum number of steps to
                evaluate. Defaults to 100.
            done_type (str, optional): the type of the done. Defaults to
                'fail'.
            terminal_type (str, optional): the type of the terminal cost used
                in the backup critic update. Defaults to 'const'.
        """
        self.state_bound = 2.
        if task is None:
            task = {}
        super(VanillaEnv, self).__init__(
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
            sample_inside_tar=sample_inside_tar,
            max_step_train=max_step_train,
            max_step_eval=max_step_eval,
            done_type=done_type,
            terminal_type=terminal_type,
            reward_type=reward_type,
            reward_goal=reward_goal,
            reward_obs=reward_obs,
            reward_wander=reward_wander,
        )

        # Defines environment dimensions.
        self.bounds = np.array([[0., self.state_bound],
                                [-self.state_bound / 2, self.state_bound / 2],
                                [0, 2 * np.pi]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]
        self.wall_height = 1.
        self.wall_thickness = 0.05
        self.has_ceiling = has_ceiling

        # Defines robot dimension and dynamics.
        self.robot_dim = [
            0.10, 0.06, 0.20
        ]  # half dims, only for visualization - for collision, assume point
        self.robot_com_height = self.robot_dim[2]
        self.dt = 0.1
        self._init_goal_dist = 2

        # Defines observation info.
        self.camera_height = 0.2  # cannot be too low, otherwise bad depth
        self.camera_x_offset = -self.robot_dim[0] / 2  # at back
        self.camera_aspect = 1
        self.camera_fov = 120.0
        self.camera_tilt = 0

        # Defines colors.
        self.ground_rgba = [0.9, 0.9, 0.9, 1.0]
        if uniform_wall_color:
            self.left_wall_rgba = [0.5, 0.5, 0.5, 1.0]
            self.back_wall_rgba = [0.5, 0.5, 0.5, 1.0]
            self.right_wall_rgba = [0.5, 0.5, 0.5, 1.0]
            self.front_wall_rgba = [0.5, 0.5, 0.5, 1.0]
        else:  # different greyscale
            self.left_wall_rgba = [0.1, 0.1, 0.1, 1.0]
            self.back_wall_rgba = [0.3, 0.3, 0.3, 1.0]
            self.right_wall_rgba = [0.5, 0.5, 0.5, 1.0]
            self.front_wall_rgba = [0.7, 0.7, 0.7, 1.0]
        self.goal_rgba = [0.0, 1.0, 0.0, 1.0]  # green

        # Defines robot states.
        self._init_state = np.array([0.1, 0., 0.])
        if fixed_init:
            self.visual_initial_states = np.array([[0.1, 0.0]])
            self.theta_init_range = [-np.pi / 3, np.pi / 3]
            self.y_init_range = [-0.5, 0.5]
        else:
            self.visual_initial_states = np.array([[0.3, 0.7], [1., -0.5],
                                                   [1.5, 0.], [0.5, 0.]])

        # Extracts task info - overwritten by reset.
        self._obs_id_all = []
        self.set_default_task(task)

        # Defines action space of the controller models.
        self.action_lim = np.float32(np.array([1., 1.]))
        self.thetadot_range_model = [-1., 1.]
        self.xdot_range_perf_model = [0.2, 1.0]
        self.xdot_range_backup_model = [0.2, 0.5]

        # Defines action space of the real environment.
        self.thetadot_range = [-1., 1.]
        self.xdot_range_perf = [0.2, 1.0]
        self.xdot_range_backup = [0.2, 0.5]

        # Fixs seed.
        self.seed(0)

    @property
    def env_type(self):
        return 'vanilla'

    @property
    def state_dim(self):
        return 3

    @property
    def action_dim(self):
        return 2

    def report(self, print_obs_state=True, print_training=True, **kwargs):
        """Prints information of robot dynamics and observation.
        """
        print("\n== ENVIRONMENT INFORMATION==")
        if print_obs_state:
            print("Observation/State:")
            print(
                "  - obs shape: ({} x {} x {})".format(
                    self.num_img_channel, self.img_w, self.img_h
                )
            )
            print("  - state dimension: {:d}".format(self.state_dim))

        if print_training:
            print("Training:")
            print("  - use sparse reward: {}".format(self.sparse_reward))
            print("  - reward type: {}".format(self.reward_type))
            print("  - reward goal: {}".format(self.reward_goal))
            print("  - reward wander: {}".format(self.reward_wander))
            print("  - reward obstacle: {}".format(self.reward_obs))
            print("  - done type: {}".format(self.done_type))
            print("  - terminal type: {}".format(self.terminal_type))

        print("Dynamic parameters:")
        print(
            "  - action dimension: {:d}, dt: {:.1f}".format(
                self.action_dim, self.dt
            )
        )
        print("  - v_perf: ", self.xdot_range_perf)
        print("  - v_backup: ", self.xdot_range_backup)
        print("  - w: ", self.thetadot_range)
        print("  - yaw_range: ", self._goal_yaw_range)
        print("== END OF REPORT==\n")

    def init_pb(self):
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
            p.resetDebugVisualizerCamera(
                3.0, 180, -89, [self.state_bound / 2, 0, 0]
            )

        # Loads ground, walls
        ground_collision_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[
                self.state_bound / 2, self.state_bound / 2,
                self.wall_thickness / 2
            ]
        )
        ground_visual_id = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=self.ground_rgba, halfExtents=[
                self.state_bound / 2, self.state_bound / 2,
                self.wall_thickness / 2
            ]
        )
        self.ground_id = p.createMultiBody(
            baseMass=0,  # FIXED
            baseCollisionShapeIndex=ground_collision_id,
            baseVisualShapeIndex=ground_visual_id,
            basePosition=[self.state_bound / 2, 0, -self.wall_thickness / 2]
        )
        wall_collision_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[
                self.wall_thickness / 2, self.state_bound / 2,
                self.wall_height / 2
            ]
        )
        wall_back_visual_id = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=self.back_wall_rgba, halfExtents=[
                self.wall_thickness / 2, self.state_bound / 2,
                self.wall_height / 2
            ]
        )
        self.wall_back_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=wall_collision_id,
            baseVisualShapeIndex=wall_back_visual_id,
            basePosition=[-self.wall_thickness / 2, 0, self.wall_height / 2]
        )
        wall_left_visual_id = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=self.left_wall_rgba, halfExtents=[
                self.wall_thickness / 2, self.state_bound / 2,
                self.wall_height / 2
            ]
        )
        self.wall_left_id = p.createMultiBody(  # positive in y
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_id,
            baseVisualShapeIndex=wall_left_visual_id,
            basePosition=[
                self.state_bound / 2,
                self.state_bound/2 + self.wall_thickness/2,
                self.wall_height / 2
            ],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        wall_right_visual_id = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=self.right_wall_rgba, halfExtents=[
                self.wall_thickness / 2, self.state_bound / 2,
                self.wall_height / 2
            ]
        )
        self.wall_right_id = p.createMultiBody(  # negative in y
            baseMass=0,
            baseCollisionShapeIndex=wall_collision_id,
            baseVisualShapeIndex=wall_right_visual_id,
            basePosition=[
                self.state_bound / 2,
                -self.state_bound/2 - self.wall_thickness/2,
                self.wall_height / 2
            ],
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        wall_front_visual_id = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=self.front_wall_rgba, halfExtents=[
                self.wall_thickness / 2, self.state_bound / 2,
                self.wall_height / 2
            ]
        )
        self.wall_front_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=wall_collision_id,
            baseVisualShapeIndex=wall_front_visual_id, basePosition=[
                self.state_bound + self.wall_thickness / 2, 0,
                self.wall_height / 2
            ]
        )
        if self.has_ceiling:
            self.wall_top_id = p.createMultiBody(
                # for blocking view - same as ground
                baseMass=0,
                baseCollisionShapeIndex=ground_collision_id,
                baseVisualShapeIndex=ground_visual_id,
                basePosition=[
                    self.state_bound / 2, 0,
                    self.wall_height + self.wall_thickness / 2
                ]
            )

    def close_pb(self):
        if "door_id" in vars(self).keys():
            self._p.removeBody(self.door_id)
            del self.door_id
        if "robot_id" in vars(self).keys():
            self._p.removeBody(self.robot_id)
            del self.robot_id
        for id in self._obs_id_all:
            self._p.removeBody(id)
        self._obs_id_all = []
        self._p.disconnect()
        self._physics_client_id = -1
        del self._p

    def set_default_task(self, task):
        def_x = self.state_bound / 2
        def_c = [1.0, 0.0, 0.0, 1.0]
        self._task = task
        self._goal_loc = task.get(
            'goal_loc', np.array([self.state_bound - 0.2, 0.])
        )
        self._goal_radius = task.get('goal_radius', 0.15)
        self._num_obs = task.get('num_obs', 5)
        self._obs_dict = task.get(
            'obs_dict', [
                dict(
                    obs_type='circle', loc=[def_x, -0.6], radius=0.05,
                    rgba=def_c
                ),
                dict(
                    obs_type='circle', loc=[def_x, -0.3], radius=0.05,
                    rgba=def_c
                ),
                dict(
                    obs_type='circle', loc=[def_x, 0], radius=0.05, rgba=def_c
                ),
                dict(
                    obs_type='circle', loc=[def_x, 0.3], radius=0.05,
                    rgba=def_c
                ),
                dict(
                    obs_type='circle', loc=[def_x, 0.6], radius=0.05,
                    rgba=def_c
                )
            ]
        )
        self.grid_cells = (100, 100)
        self._goal_yaw_range = task.get(
            'goal_yaw_range', np.array([[0., 2 * np.pi]])
        )

    def reset_task(self, task):
        self._task = task
        if 'goal_loc' in task:
            self._goal_loc = task['goal_loc']
        if 'goal_radius' in task:
            self._goal_radius = task['goal_radius']
        if 'goal_yaw_range' in task:
            self._goal_yaw_range = task['goal_yaw_range']
        if 'num_obs' in task:
            self._num_obs = task['num_obs']
        if 'obs_dict' in task:
            self._obs_dict = task['obs_dict']
        if 'grid_cells' in task:
            self.grid_cells = task['grid_cells']
        if 'init_state' in task:
            self._init_state = task['init_state']
        if 'get_yaw_append' in task:
            self.get_yaw_append = task['get_yaw_append']
            if self.get_yaw_append:
                self.yaw_append = task['yaw_append']
        if 'xdot_range_perf' in task:
            self.xdot_range_perf = task['xdot_range_perf']
        if 'xdot_range_backup' in task:
            self.xdot_range_backup = task['xdot_range_backup']
        if 'thetadot_range' in task:
            self.thetadot_range = task['thetadot_range']

        # Reset obstacles
        self.reset_obstacles()

        # Reset goal
        self.reset_goal()

        # Update occupancy map
        if 'occupancy_map' in task:
            self.occupancy_map = task['occupancy_map']
        else:
            self.get_occupancy_map(grid_cells=self.grid_cells)

    def reset_obstacles(self):
        # Remove existing ones
        for id in self._obs_id_all:
            self._p.removeBody(id)
        self._obs_id_all = []

        # Load
        for obs_ind in range(self._num_obs):
            obs_info = self._obs_dict[obs_ind]
            loc = obs_info['loc']
            rgba = obs_info['rgba']
            if obs_info['obs_type'] == 'circle':
                radius = obs_info['radius']
                theta = 0
                obs_collision_id = self._p.createCollisionShape(
                    self._p.GEOM_CYLINDER, radius=radius,
                    height=self.wall_height
                )
                obs_visual_id = self._p.createVisualShape(
                    self._p.GEOM_CYLINDER, rgbaColor=rgba, radius=radius,
                    length=self.wall_height
                )
            else:
                width = obs_info['width']
                height = obs_info['height']
                theta = obs_info['theta']
                obs_collision_id = self._p.createCollisionShape(
                    self._p.GEOM_BOX,
                    halfExtents=[width, height, self.wall_height / 2]
                )
                obs_visual_id = self._p.createVisualShape(
                    self._p.GEOM_BOX, rgbaColor=rgba,
                    halfExtents=[width, height, self.wall_height / 2]
                )

            obs_id = self._p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=obs_collision_id,
                baseVisualShapeIndex=obs_visual_id,
                basePosition=np.append(loc, self.wall_height / 2),
                baseOrientation=self._p.getQuaternionFromEuler([0, 0, theta])
            )
            self._obs_id_all += [obs_id]

    def reset_goal(self):
        if "door_id" in vars(self).keys():
            self._p.removeBody(self.door_id)
        door_visual_id = self._p.createVisualShape(
            self._p.GEOM_BOX, rgbaColor=self.goal_rgba,
            halfExtents=[0.01, self._goal_radius, self.wall_height / 2]
        )
        self.door_id = self._p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=door_visual_id,
            basePosition=[
                self.state_bound - 0.01, self._goal_loc[1],
                self.wall_height / 2
            ],
        )

    def reset_robot(self, state):
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
                baseMass=0,  # no dynamics
                baseVisualShapeIndex=robot_visual_id,
                basePosition=np.append(state[:2], self.robot_com_height),
                baseOrientation=self._p.getQuaternionFromEuler([
                    0, 0, state[2]
                ])
            )

    def reset(self, random_init=False, state_init=None, task=None):
        if self._physics_client_id < 0:
            self.init_pb()

        if state_init is not None:
            self._state = state_init
        elif random_init:
            self._state = self.sample_state(
                self.sample_inside_obs, self.sample_inside_tar
            )
        elif self.fixed_init:  #! no sampling because of ps training
            self._state = self._init_state.copy()
            # self._state[-1] = self.rng.uniform(
            #     self.theta_init_range[0], self.theta_init_range[1]
            # )
            # self._state[1] = self.rng.uniform(
            #     self.y_init_range[0], self.y_init_range[1]
            # )
        else:
            raise ValueError("Need one method to get an initial state")

        # Update init
        self._init_state = self._state.copy()

        self.step_elapsed = 0  # Reset timer

        if task is not None:  # we at least reset to default_task in init
            self._task = task
            self.reset_task(task)

        self.reset_robot(self._state.copy())  # Reset robot

        return self._get_obs(self._state)

    def get_occupancy_map(
        self, grid_cells=(100, 100), add_init=False, add_goal=True,
        get_g_x=False, get_l_x=False
    ):

        # channel 1: obstacle; channel 2: goal; channel 3: state/traj.
        num_channel = 2
        if add_init:
            num_channel = 3
        occ_map = np.zeros((num_channel,) + grid_cells, dtype='uint8')
        if get_g_x or get_l_x:
            v_x_map = np.empty(grid_cells)

        it = np.nditer(occ_map[0], flags=['multi_index'])

        while not it.finished:
            idx = it.multi_index
            state = index_to_state(grid_cells, self.bounds, idx, mode='center')
            g_x = self.safety_margin(state)
            l_x = self.target_margin(state)

            if g_x > 0:
                occ_map[0][idx] = 255
            elif add_goal:
                if l_x <= 0:
                    occ_map[1][idx] = 255

            if get_g_x:
                if get_l_x:
                    v_x_map[idx] = max(g_x, l_x)
                else:
                    v_x_map[idx] = g_x
            elif get_l_x:
                v_x_map[idx] = l_x
            it.iternext()

        if add_init:
            idx = state_to_index(grid_cells, self.bounds, self._state[:2])
            occ_map[2][idx] = 255
        self.occupancy_map = occ_map
        self.grid_cells = grid_cells
        if get_g_x or get_l_x:
            return occ_map, v_x_map
        return occ_map

    def sample_state(
        self, sample_inside_obs=False, sample_inside_tar=True, theta=None,
        fixed_init=False, **kwargs
    ):
        """
        Sample a state for robot.

        Args:
            sample_inside_obs (bool, optional): whether to sample the initial
                state inside the obstacle. Defaults to False.
            sample_inside_tar (bool, optional): whether to sample the initial
                state inside the target. Defaults to True.
            theta (float, optional): the angle of the robot. Defaults to None.

        Returns:
            np.array: the sampled state.
        """
        if fixed_init:
            return self._task['init_state'].copy()
        else:
            if theta is not None:
                theta_rnd = theta
            else:
                theta_rnd = 2.0 * np.pi * self.rng.uniform()

            flag = True
            low = self.low[:2]
            high = self.high[:2]
            while flag:
                rnd_state = self.rng.uniform(low=low, high=high)
                l_x = self.target_margin(rnd_state)
                g_x = self.safety_margin(rnd_state)

                if (not sample_inside_obs) and (g_x > 0):
                    flag = True
                elif (not sample_inside_tar) and (l_x <= 0):
                    flag = True
                else:
                    flag = False
            x_rnd, y_rnd = rnd_state

            return np.array([x_rnd, y_rnd, theta_rnd])

    def integrate_forward(self, state, v, w, num_pt=10):
        """
        Integrate the dynamics forward by one step.

        Args:
            state: x, y, theta.
            w: angular speed.

        Returns:
            State variables (x,y,theta) integrated one step forward in time.
        """
        dt = self.dt / num_pt
        x, y, theta = state
        delta_x = v * np.cos(theta) * dt
        delta_y = v * np.sin(theta) * dt
        for i in range(num_pt):
            x = x + delta_x
            y = y + delta_y
            g_x = self.safety_margin(np.array([x, y]))
            if g_x > 0:
                break
        theta_new = np.mod(theta + w * dt * (i+1), 2 * np.pi)
        state = np.array([x, y, theta_new])

        return state

    def get_control(self, action):
        """
        Transforms actions from the neural network to the real control inputs.

        Args:
            action (np.ndarray): (2, ), each element is within [-1, 1].

        Returns:
            float: linear velocity.
            float: angular velocity.
        """
        if action[-1] == 1:
            xdot_range_model = self.xdot_range_perf_model
            xdot_range = self.xdot_range_perf
        else:
            xdot_range_model = self.xdot_range_backup_model
            xdot_range = self.xdot_range_backup

        # Maps [-1, 1] to the action space of the policy.
        v = scale_and_shift(action[0], [-1, 1], xdot_range_model)

        # Clips the actions based on the env constraints. There can be
        # discrepancy between the model and env.
        v = max(min(v, xdot_range[1]), xdot_range[0])

        # w = scale_and_shift(action[1], [-1, 1], self.thetadot_range_model)
        w = max(min(action[1], self.thetadot_range[1]), self.thetadot_range[0])

        return v, w

    def move_robot(self, action, state):
        """
        Moves the robot by appling given action form the state .

        Args:
            action (np.ndarray): the action to be applied.
            state (np.ndarray): the state of the robot.
        """
        assert action.ndim == state.ndim, "the dimensions do not match!"
        if action.ndim == 2 and action.shape[0] > 1:
            raise ValueError("move_robot only supports one action right now!")
        # add_extra_dim = False
        if action.ndim == 2:
            action = action[0, :]
            state = state[0, :]
            # add_extra_dim = True

        # print(action, state)

        v, w = self.get_control(action)
        state = self.integrate_forward(state, v, w)
        self._p.resetBasePositionAndOrientation(
            self.robot_id, posObj=np.append(state[:2], self.robot_com_height),
            ornObj=self._p.getQuaternionFromEuler([0, 0, state[2]])
        )
        # if add_extra_dim:
        #     state = state[np.newaxis, :]
        return state

    def step(self, action):
        # Move car in simulation
        self._state = self.move_robot(action, self._state)

        # = `l_x` and `g_x` signal
        info = self._get_info(self._state)
        l_x = info['l_x']
        g_x = info['g_x']
        boundary_margin = info['boundary_margin']
        obstacle_margin = info['obstacle_margin']
        target_yaw_margin = info['target_yaw_margin']

        # prevent bad image at the boundary - small value to buffer
        # The robot is safe iff the safety margin is negative
        fail = ((boundary_margin > -self.boundary_buffer)
                or (obstacle_margin > -self.obs_buffer))
        success = l_x <= 0

        # = `reward` signal
        dist_to_goal_center = np.linalg.norm(self._state[:2] - self._goal_loc)
        if self.sparse_reward:
            # small penalty for wandering around
            reward = self.reward_wander

            # large reward for reaching target
            if success:
                reward = self.reward_goal
        else:
            # dense `reward`
            if success:
                reward = self.reward_goal
            else:
                coeff = 1.0
                if target_yaw_margin > 0:
                    coeff = 0.5
                reward = coeff * (self.reward_wander - dist_to_goal_center)

        if self.reward_type == 'all' and fail:
            # if type is 'all', the performance policy cares collision.
            reward = self.reward_obs

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

        # = `info` signal
        # for recovery RL
        binary_cost = 0
        if fail:
            binary_cost = 1

        # Ideally, reach-avoid or safety Bellman equation wants to have
        # trajectories penetrating deep inside the obstacle to propagate the
        # positive value. But, in real world, the robot might not be allowed to
        # do so, so we assign a constant for terminal g_x.
        if self.done_type == 'fail' and self.terminal_type == 'const' and fail:
            g_x = self.g_x_fail

        info = {
            'task': self._task,
            'state': self._state,
            'g_x': g_x,
            'l_x': l_x,
            'k_x': np.linalg.norm(self._state[:2] - self._init_state[:2]),
            'binary_cost': binary_cost,
        }

        return self._get_obs(self._state), reward, done, info

    def safety_margin(self, state, return_boundary=False):
        """
        Computes the margin (e.g. distance) between state and failue set.

        Args:
            state: consisting of [x, y, theta].

        Returns:
            g(state): safety margin, positive value suggests inside of the
                failure set.
        """
        s = state[:2]

        x, y = (self.bounds[:, 0] + self.bounds[:, 1])[:2] / 2.0
        w, h = (self.bounds[:, 1] - self.bounds[:, 0])[:2] / 2.0
        x_y_w_h_theta = [x, y, w, h, 0]
        boundary_margin = self._calculate_margin_rect(
            s, x_y_w_h_theta, negative_inside=True
        )
        obstacle_list = []
        for obs_info in self._obs_dict:
            loc = obs_info['loc']
            if obs_info['obs_type'] == 'circle':
                radius = obs_info['radius']
                obstacle_list += [
                    self._calculate_margin_circle(
                        s, [loc, radius], negative_inside=False
                    )
                ]
            else:
                width = obs_info['width']
                height = obs_info['height']
                theta = obs_info['theta']
                x_y_w_h_theta = [loc[0], loc[1], width, height, theta]
                obstacle_list += [
                    self._calculate_margin_rect(
                        s, x_y_w_h_theta, negative_inside=False
                    )
                ]

        obstacle_margin = np.max(obstacle_list)
        safety_margin = max(obstacle_margin, boundary_margin)

        # Smooth
        safety_margin += self.obs_buffer

        if return_boundary:
            return safety_margin, boundary_margin, obstacle_margin
        else:
            return safety_margin

    # Visualization utils.
    def visualize(
        self, q_func, policy, mode, use_rnd_traj=False, num_rnd_traj=10,
        end_type='TF', vmin=-1, vmax=1, nx=51, ny=51, cmap='seismic',
        labels=None, bool_plot=False, plot_v=True, normalize_v=False,
        plot_contour=True, traj_size=0, frame_skip=0, latent_dist=None,
        task=None, revert_task=True, shield=False, backup=None,
        shield_dict=None, **kwargs
    ):
        """
        visualize
        Args:
            q_func (object): agent's Q-network.
            policy (func): agent's policy.
            use_rnd_traj (bool, optional): random initialization or not.
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
        theta_list = [-np.pi / 4, 0, np.pi / 4]
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax_list = [ax1, ax2, ax3]

        if revert_task:
            prev_task = self._task

        if task is not None:
            self.reset_task(task)

        # For get_value()
        latent = None
        if latent_dist is not None:
            latent = latent_dist.mean.unsqueeze(0)

        for _, (ax, theta) in enumerate(zip(ax_list, theta_list)):
            # if i == len(theta_list) - 1:
            #     plot_cbar = True
            # else:
            #     plot_cbar = False
            plot_cbar = True

            # == Plot failure / target set ==
            if end_type == 'TF':
                self.plot_target_failure_set(ax)
            else:
                self.plot_failure_set(ax)

            # == Plot V ==
            if plot_v:
                self.plot_v_values(
                    q_func, fig, ax, theta=theta, bool_plot=bool_plot,
                    plot_cbar=plot_cbar, vmin=vmin, vmax=vmax, nx=nx, ny=ny,
                    cmap=cmap, normalize_v=normalize_v,
                    plot_contour=plot_contour, alpha=0.5, latent=latent,
                    traj_size=traj_size, fontsize=16
                )

            # == Plot Trajectories ==
            visual_initial_states = np.tile(
                self.visual_initial_states, (self.num_traj_per_visual_init, 1)
            )
            thetas = theta * np.ones(shape=(visual_initial_states.shape[0], 1))
            states = np.concatenate((visual_initial_states, thetas), axis=1)
            sim_traj_args = dict(
                end_type=end_type, latent_dist=latent_dist,
                traj_size=traj_size, frame_skip=frame_skip, shield=shield,
                backup=backup, shield_dict=shield_dict
            )
            if use_rnd_traj:
                sample_args = dict(
                    theta=theta, sample_inside_obs=False,
                    sample_inside_tar=True
                )
                self.plot_trajectories(
                    policy, mode, ax, num_rnd_traj=num_rnd_traj,
                    sample_args=sample_args, sim_traj_args=sim_traj_args
                )
            else:
                self.plot_trajectories(
                    policy, mode, ax, states=states,
                    sim_traj_args=sim_traj_args
                )

            # == Formatting ==
            self.plot_formatting(ax, labels=labels)
            fig.tight_layout()

            ax.set_xlabel(
                r'$\theta={:.0f}^\circ$'.format(theta * 180 / np.pi),
                fontsize=28
            )

        if revert_task:
            self.reset_task(prev_task)

        return fig

    def plot_failure_set(self, ax, c_c='m', lw=3, zorder=0):
        for obs_info in self._obs_dict:
            loc = obs_info['loc']
            if obs_info['obs_type'] == 'circle':
                radius = obs_info['radius']
                plot_circle(loc, radius, ax, c=c_c, lw=lw, zorder=zorder)
            else:
                width = obs_info['width']
                height = obs_info['height']
                theta = obs_info['theta']
                plot_rect(
                    loc, width, height, theta, ax, c=c_c, lw=lw, zorder=zorder
                )

    def plot_target_set(self, ax, c_t='y', lw=3, zorder=0):
        plot_circle(
            self._goal_loc, self._goal_radius, ax, c=c_t, lw=lw, zorder=zorder
        )

    def plot_target_failure_set(self, ax, c_c='m', c_t='y', lw=3, zorder=0):
        """
        Plot target and obstacles.

        Args:
            ax (matplotlib.axes.Axes).
            c_c (str, optional): the color of constraint set boundary.
                Defaults to 'm'.
            c_t (str, optional): the color of target set boundary.
                Defaults to 'y'.
            lw (int, optional): the lw of the boundaries. Defaults to 3.
            zorder (int, optional): the graph oder of the boundaries.
                Defaults to 0.
        """
        self.plot_failure_set(ax, c_c, lw, zorder)
        self.plot_target_set(ax, c_t, lw, zorder)
