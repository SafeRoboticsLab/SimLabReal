import numpy as np
import random
import pybullet as pb
from pybullet_utils import bullet_client as bc
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from abc import abstractmethod
import torch

# import os
# os.sys.path.append(os.path.join(os.getcwd(), '.'))
from safe_rl.util_geom import euler2rot
from gym_reachability.gym_reachability.envs.env_utils import plot_arc, plot_circle


def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


class NavigationObsPBEnv(gym.Env):
    """Simple 2D navigation with obstacle using PyBullet. No dynamics/collision
        simulation. If PyBullet is not fast enough, we can use a renderer
        instead.
    State: x,y,theta of the car
    Dynamics: Dubins car, fixed forward speed v, turning rate {w,0,-w},
        thus discrete control
    Right now there is a single obstacle at the center of the map. Info from
        step() stores distance to obstacle boundary (g) and distance to goal (l)
    """
    def __init__(self,  task={},
                        img_H=128,
                        img_W=128,
                        useRGB=True,
                        render=True,
                        sample_inside_obs=False,
                        uniformWallColor=False,
                        doneType='fail'):
        """
        __init__: initialization

        Args:
            task (dict, optional): task information dictionary. Defaults to {}.
            img_H (int, optional): height of the observation. Defaults to 96.
            img_W (int, optional): width of the observation.. Defaults to 96.
            render (bool, optional): use pb.GUI if True. Defaults to True.
        """
        super(NavigationObsPBEnv, self).__init__()

        # Define dimensions
        self.state_bound = 2.
        self.bounds = np.array([[0., self.state_bound],
                                [-self.state_bound/2, self.state_bound/2],
                                [0, 2*np.pi]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]
        self.wall_height = 1.
        self.wall_thickness = 0.05
        self.car_dim = [0.04, 0.02, 0.01]	# half dims, only for visualization
        self.camera_height = 0.2	# cannot be too low, otherwise bad depth
        self.sample_inside_obs=sample_inside_obs
        self.sample_inside_tar=True

        # Set up observation and action space for Gym
        self.img_H = img_H
        self.img_W = img_W
        self.useRGB = useRGB
        if useRGB:
            num_img_channel = 3 # RGB
        else:
            num_img_channel = 1 # D only
        self.observation_space = gym.spaces.Box(
            low=np.float32(0.),
            high=np.float32(1.),
            shape=(num_img_channel, img_H, img_W))
        self.action_lim = np.float32(np.array([1.])) #! action_space is defined in the child class

        # Color
        self.ground_rgba = [0.9, 0.9, 0.9, 1.0]
        if uniformWallColor:
            self.left_wall_rgba = [0.5, 0.5, 0.5, 1.0]
            self.back_wall_rgba = [0.5, 0.5, 0.5, 1.0]
            self.right_wall_rgba = [0.5, 0.5, 0.5, 1.0]
            self.front_wall_rgba = [0.5, 0.5, 0.5, 1.0]
        else:   # different greyscale
            self.left_wall_rgba = [0.1, 0.1, 0.1, 1.0]
            self.back_wall_rgba = [0.3, 0.3, 0.3, 1.0]
            self.right_wall_rgba = [0.5, 0.5, 0.5, 1.0]
            self.front_wall_rgba = [0.7, 0.7, 0.7, 1.0]
        self.obs_rgba = [1.0, 0.0, 0.0, 1.0]    # red
        self.goal_rgba  = [0.0, 1.0, 0.0, 1.0]  # green

        # Car initial x/y/theta
        self.car_init_state = np.array([0.1, 0., 0.])
        self.visual_initial_states = np.array([ [ 0.3,  0.7],
                                                [ 1.,  -0.5],
                                                [ 1.5,  0. ],
                                                [ 0.5,  0. ]])
        # Car dynamics
        self.state_dim = 3
        self.action_dim = 1
        self.v = 0.2
        self.dt = 0.1
        self.doneType = doneType

        # Extract task info
        self._task = task
        self._goal_loc = task.get('goal_loc', np.array([self.state_bound-0.2, 0.]))
        self._goal_radius = task.get('goal_radius', 0.15)
        self._obs_loc  = task.get('obs_loc', np.array([self.state_bound/2, 0]))
        self._obs_radius = task.get('obs_radius', 0.3)
        self._obs_buffer = 0.2 # no cost if outside buffer
        self._boundary_buffer = 0.02    # not too close to boundary
        self.obs_reward_scale = 1

        # Set up PyBullet parameters
        self._renders = render
        self._physics_client_id = -1


    def seed(self, seed=None):
        self.seed_val = seed
        self.action_space.seed(self.seed_val)
        np.random.seed(self.seed_val)
        random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # self.np_random, seed = gym.utils.seeding.np_random(seed)
        # return [seed]


    # TODO: call this in multi-env setting
    def reset_task(self, task):
        self._task = task
        self._goal_loc = task['goal_loc']
        self._goal_radius = task['goal_radius']
        self._obs_loc = task['obs_loc']
        self._obs_radius = task['obs_radius']


    def reset(self, random_init=True, state_init=None):
        if random_init:
            self._state = self.sample_state(self.sample_inside_obs,
                                            self.sample_inside_tar)
        elif state_init is not None:
            self._state = state_init
        else:
            self._state = self.car_init_state

        # Start PyBullet session if first time
        # print("----------- reset simulation ---------------")
        if self._physics_client_id < 0:
            # print("------------ start pybullet ----------------")
            if self._renders:
                self._p = bc.BulletClient(connection_mode=pb.GUI)
            else:
                self._p = bc.BulletClient()
            self._physics_client_id = self._p._client
            p = self._p
            p.resetSimulation()
            p.setTimeStep(self.dt)
            p.setRealTimeSimulation(0)
            p.setGravity(0, 0, -9.8)
            if self._renders:
                p.resetDebugVisualizerCamera(3.0, 180, -89,
                    [self.state_bound/2, 0, 0])

            # Load ground, walls
            ground_collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[	self.state_bound/2,
                                self.state_bound/2,
                                self.wall_thickness/2])
            ground_visual_id = p.createVisualShape(
                p.GEOM_BOX,
                rgbaColor=self.ground_rgba,
                halfExtents=[	self.state_bound/2,
                                self.state_bound/2,
                                self.wall_thickness/2])
            self.ground_id = p.createMultiBody(
                baseMass=0,	# FIXED
                baseCollisionShapeIndex=ground_collision_id,
                baseVisualShapeIndex=ground_visual_id,
                basePosition=[	self.state_bound/2,
                                0,
                                -self.wall_thickness/2])
            wall_collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[	self.wall_thickness/2,
                                self.state_bound/2,
                                self.wall_height/2])
            wall_back_visual_id = p.createVisualShape(
                p.GEOM_BOX,
                rgbaColor=self.back_wall_rgba,
                halfExtents=[	self.wall_thickness/2,
                                self.state_bound/2,
                                self.wall_height/2])
            self.wall_back_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision_id,
                baseVisualShapeIndex=wall_back_visual_id,
                basePosition=[	-self.wall_thickness/2,
                                0,
                                self.wall_height/2])
            wall_left_visual_id = p.createVisualShape(
                p.GEOM_BOX,
                rgbaColor=self.left_wall_rgba,
                halfExtents=[	self.wall_thickness/2,
                                self.state_bound/2,
                                self.wall_height/2])
            self.wall_left_id = p.createMultiBody(	# positive in y
                baseMass=0,
                baseCollisionShapeIndex=wall_collision_id,
                baseVisualShapeIndex=wall_left_visual_id,
                basePosition=[	self.state_bound/2,
                                self.state_bound/2+self.wall_thickness/2,
                                self.wall_height/2],
                baseOrientation=p.getQuaternionFromEuler([0,0,np.pi/2]))
            wall_right_visual_id = p.createVisualShape(
                p.GEOM_BOX,
                rgbaColor=self.right_wall_rgba,
                halfExtents=[	self.wall_thickness/2,
                                self.state_bound/2,
                                self.wall_height/2])
            self.wall_right_id = p.createMultiBody(	# negative in y
                baseMass=0,
                baseCollisionShapeIndex=wall_collision_id,
                baseVisualShapeIndex=wall_right_visual_id,
                basePosition=[	self.state_bound/2,
                                -self.state_bound/2-self.wall_thickness/2,
                                self.wall_height/2],
                baseOrientation=p.getQuaternionFromEuler([0,0,np.pi/2]))
            wall_front_visual_id = p.createVisualShape(
                p.GEOM_BOX,
                rgbaColor=self.front_wall_rgba,
                halfExtents=[	self.wall_thickness/2,
                                self.state_bound/2,
                                self.wall_height/2])
            self.wall_front_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision_id,
                baseVisualShapeIndex=wall_front_visual_id,
                basePosition=[	self.state_bound+self.wall_thickness/2,
                                0,
                                self.wall_height/2])
            self.wall_top_id = p.createMultiBody(	# for blocking view - same as ground
                baseMass=0,
                baseCollisionShapeIndex=ground_collision_id,
                baseVisualShapeIndex=ground_visual_id,
                basePosition=[	self.state_bound/2,
                                0,
                                self.wall_height+self.wall_thickness/2])

            # Obstacle
            obs_collision_id = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=self._obs_radius,
                height=self.wall_height)
            obs_visual_id = p.createVisualShape(
                p.GEOM_CYLINDER,
                rgbaColor=self.obs_rgba,
                radius=self._obs_radius,
                length=self.wall_height)
            self.obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=obs_collision_id,
                baseVisualShapeIndex=obs_visual_id,
                basePosition=np.append(self._obs_loc, self.wall_height/2))

            # Door - behind the virtual target
            door_visual_id = p.createVisualShape(
                p.GEOM_BOX,
                rgbaColor=self.goal_rgba,
                halfExtents=[	0.01,
                                self._goal_radius,
                                self.wall_height/2])
            self.door_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=door_visual_id,
                basePosition=[self.state_bound-0.01,
                            self._goal_loc[1],
                            self.wall_height/2])

            # Set up car if visualizing in GUI
            if self._renders:
                car_collision_id = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=self.car_dim)
                car_visual_id = -1
                car_visual_id = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=self.car_dim,
                    rgbaColor=[0,0,1,1])
                self.car_id = p.createMultiBody(
                    baseMass=0, # no dynamics
                    baseCollisionShapeIndex=car_collision_id,
                    baseVisualShapeIndex=car_visual_id,
                    basePosition=np.append(self._state[:2], 0.03),
                    baseOrientation=p.getQuaternionFromEuler([0,0,0]))
            p.stepSimulation()
        else:
            if self._renders:
                self._p.resetBasePositionAndOrientation(self.car_id,
                    np.append(self._state[:2], 0.03),
                    self._p.getQuaternionFromEuler([0,0,self._state[2]]))
        return self._get_obs(self._state)


    def sample_state(self, sample_inside_obs=False, sample_inside_tar=True, theta=None):
        # random sample `theta`
        if theta is not None:
            theta_rnd = theta
        else:
            theta_rnd = 2.0 * np.pi * np.random.uniform()

        # random sample [`x`, `y`]
        flag = True
        low = self.low[:2]
        high = self.high[:2]
        while flag:
            rnd_state = np.random.uniform(low=low, high=high)
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


    def integrate_forward(self, state, w):
        """ Integrate the dynamics forward by one step.

        Args:
            state: x, y, theta.
            w: angular speed.

        Returns:
            State variables (x,y,theta) integrated one step forward in time.
        """

        x, y, theta = state
        x_new = x + self.v*np.cos(theta)*self.dt
        y_new = y + self.v*np.sin(theta)*self.dt
        theta_new = np.mod(theta + w*self.dt, 2*np.pi)
        state = np.array([x_new, y_new, theta_new])

        return state

    @abstractmethod
    def getTurningRate(self, action):
        raise NotImplementedError


    def step(self, action):
        # Determine turning rate
        w = self.getTurningRate(action)

        #= Dynamics
        self._state = self.integrate_forward(self._state, w)

        # Move car in simulation - not necessary if not visualizing in GUI -
        # since all we need from PyBullet is simulation of the camera and the
        # obstacle field
        if self._renders:
            x_new, y_new, theta_new = self._state
            self._p.resetBasePositionAndOrientation(self.car_id,
                                [x_new,y_new,0.03],
                                self._p.getQuaternionFromEuler([0,0,theta_new]))

        #= `l_x` and `g_x` signal
        l_x = self.target_margin(self._state)
        g_x, boundary_margin = self.safety_margin(self._state, return_boundary=True)
        fail = g_x >= -0.01 # prevent bad image at the boundary - small value to buffer
        success = l_x <= 0

        #= `reward` signal
        # reward = 4
        # dist_to_goal_center = np.linalg.norm(self._state[:2] - self._goal_loc)
        # # reward_goal = -dist_to_goal_center
        # reward -= dist_to_goal_center
        # dist_to_obs_center = np.linalg.norm(self._state[:2] - self._obs_loc)
        # dist_to_obs_boundary = dist_to_obs_center-self._obs_radius
        # if dist_to_obs_center < self._obs_radius:
        #     # reward_obs = -1.1
        #     reward = 0
        # elif dist_to_obs_center < (self._obs_radius+self._obs_buffer):
        #     # reward_obs = - (1 - dist_to_obs_boundary/self._obs_buffer)
        #     reward -= (1-dist_to_obs_boundary/self._obs_buffer)
        # if boundary_margin > -self._boundary_buffer:
        #     reward -= (1+boundary_margin/self._boundary_buffer)

        # Small penalty for wandering around
        reward = -0.01
        
        # Large reward for reaching target
        dist_to_goal_center = np.linalg.norm(self._state[:2] - self._goal_loc)
        if dist_to_goal_center < self._goal_radius:
            # reward += (0.2-dist_to_goal_center)/0.2
            reward = 1

        # Large penalty for reaching obstacle or boundary
        dist_to_obs_center = np.linalg.norm(self._state[:2] - self._obs_loc)
        if dist_to_obs_center < self._obs_radius:
            reward -= 1.0
        # elif dist_to_obs_center < (self._obs_radius+self._obs_buffer):
        #     reward -= 0.1*(1-dist_to_obs_boundary/self._obs_buffer)
        if boundary_margin > -self._boundary_buffer:
            # reward -= 0.1*(1+boundary_margin/self._boundary_buffer)
            reward -= 1.0

        #= `done` signal
        if self.doneType == 'end':
            done = not self.check_within_bounds(self._state)
        elif self.doneType == 'fail':
            done = fail
        elif self.doneType == 'TF':
            done = fail or success
        else:
            raise ValueError("invalid doneType")

        # TODO: Tuning
        if done and self.doneType == 'fail':
            g_x = 1

        return self._get_obs(self._state), reward, done, {'task': self._task,
            'state': self._state, 'g_x': g_x, 'l_x': l_x}


    def _get_obs(self, state):
        """
        _get_obs: get RGB or depth image given a state

        Args:
            state (np.ndarray): (x, y, yaw)

        Returns:
            np.ndarray: RGB or depth image, of the shape (C, H, W)
        """
        # State
        x, y, yaw = state
        rot_matrix = euler2rot([yaw,0,0])
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (1, 0, 0) # x-axis
        init_up_vector = (0, 0, 1) # z-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        cam_pos = np.array([x,y,self.camera_height])	# top of car
        view_matrix = self._p.computeViewMatrix(cam_pos,
                                        cam_pos + 0.1 * camera_vector,
                                        up_vector)

        # Get Image
        far	= 1000.0
        near = 0.01
        aspect = 1.
        projection_matrix = self._p.computeProjectionMatrixFOV(
            fov=120.0, aspect=aspect, nearVal=near, farVal=far)
        _, _, rgbImg, depth, _ = self._p.getCameraImage(self.img_H, self.img_W,
            view_matrix, projection_matrix,
            flags=self._p.ER_NO_SEGMENTATION_MASK)
        depth = np.reshape(depth, (1, self.img_H, self.img_W))
        depth = far*near/(far - (far - near)*depth)
        if self.useRGB:
            rgb = rgba2rgb(rgbImg).transpose(2,0,1)/255  # CHW
            # return np.concatenate((rgb, depth))
            return rgb
        else:
            return depth


    def set_visual_initial_states(self, states):
        assert states.shape[1] == 2 or states.shape[1] == 3,\
            'The shape of states is not correct, the second dim should be 2 or 3.'
        self.visual_initial_states = states


    #== GETTER ==
    def report(self):
        print("Dynamic parameters:")
        print("- cons: {:.1f}, tar: {:.1f}".format(
            self._obs_radius, self._goal_radius))
        print("- v: {:.1f}, w_max: {:.1f}, dt: {:.1f}".format(
            self.v, self.action_lim[0], self.dt))


    def get_axes(self):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
        """
        aspect_ratio = (self.bounds[0,1]-self.bounds[0,0])/(self.bounds[1,1]-self.bounds[1,0])
        axes = np.array([self.bounds[0,0], self.bounds[0,1], self.bounds[1,0], self.bounds[1,1]])
        return [axes, aspect_ratio]


    def get_warmup_examples(self, num_warmup_samples=100):
        heuristic_v = np.zeros((num_warmup_samples, 1))
        states = np.zeros(shape=(num_warmup_samples,) + self.observation_space.shape)

        for i in range(num_warmup_samples):
            _state = self.sample_state(self.sample_inside_obs,
                                            self.sample_inside_tar)
            l_x = self.target_margin(_state)
            g_x = self.safety_margin(_state)
            heuristic_v[i,:] = np.maximum(l_x, g_x)
            states[i] = self._get_obs(_state)

        return states, heuristic_v


    def get_value(self, q_func, device, theta, nx=101, ny=101):
        """
        get_value: get the state values given the Q-network. We fix the heading
            angle of the car to `theta`.

        Args:
            q_func (object): agent's Q-network.
            device (str): agent's device.
            theta (float): the heading angle of the car.
            nx (int, optional): # points in x-axis. Defaults to 101.
            ny (int, optional): # points in y-axis. Defaults to 101.

        Returns:
            np.ndarray: values
        """
        v = np.zeros((nx, ny))
        it = np.nditer(v, flags=['multi_index'])
        xs = np.linspace(self.bounds[0,0], self.bounds[0,1], nx)
        ys = np.linspace(self.bounds[1,0], self.bounds[1,1], ny)
        while not it.finished:
            idx = it.multi_index
            x = xs[idx[0]]
            y = ys[idx[1]]

            # getCameraImage somehow hangs at the walls
            if (abs(x) == self.state_bound or abs(x) == 0) or abs(y) == self.state_bound/2:
                v[idx] = 0
            else:
                state = np.array([x, y, theta])
                obs = self._get_obs(state)
                obsTensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
                v[idx] = q_func(obsTensor).min(dim=1)[0].cpu().detach().numpy()
            it.iternext()
        return v, xs, ys


    def check_within_bounds(self, state):
        """
        check_within_bounds

        Args:
            state (np.ndarray): (x, y, yaw)

        Returns:
            bool: True if inside the environment.
        """
        for dim, bound in enumerate(self.bounds):
            flagLow = state[dim] < bound[0]
            flagHigh = state[dim] > bound[1]
            if flagLow or flagHigh:
                return False
        return True


    @staticmethod
    def _calculate_margin_rect(s, x_y_w_h, negativeInside=True):
        x, y, w, h = x_y_w_h
        delta_x = np.abs(s[0] - x)
        delta_y = np.abs(s[1] - y)
        margin = max(delta_y - h/2, delta_x - w/2)

        if negativeInside:
            return margin
        else:
            return - margin


    @staticmethod
    def _calculate_margin_circle(s, c_r, negativeInside=True):
        center, radius = c_r
        dist_to_center = np.linalg.norm(s[:2] - center)
        margin = dist_to_center - radius

        if negativeInside:
            return margin
        else:
            return - margin


    def target_margin(self, state):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            state: consisting of [x, y, theta].

        Returns:
            l(state): target margin, negative value suggests inside of the target set.
        """
        s = state[:2]
        c_r = [self._goal_loc, self._goal_radius]
        target_margin = self._calculate_margin_circle(s, c_r, negativeInside=True)
        return target_margin


    def safety_margin(self, state, return_boundary=False):
        """ Computes the margin (e.g. distance) between state and failue set.

        Args:
            state: consisting of [x, y, theta].

        Returns:
            g(state): safety margin, positive value suggests inside of the failure set.
        """
        s = state[:2]

        x, y = (self.low + self.high)[:2] / 2.0
        w, h = (self.high - self.low)[:2]
        x_y_w_h = [x, y, w, h]
        boundary_margin = self._calculate_margin_rect(s, x_y_w_h, negativeInside=True)
        g_xList = [boundary_margin]

        c_r = [self._obs_loc, self._obs_radius]
        obs_margin = self._calculate_margin_circle(s, c_r, negativeInside=False)
        g_xList.append(obs_margin)

        safety_margin = np.max(np.array(g_xList))
        if return_boundary:
            return safety_margin, boundary_margin
        else:
            return safety_margin

    #== Trajectories Rollout ==
    def simulate_one_trajectory(self, policy, T=250, endType='TF',
            state=None, theta=np.pi/2, sample_inside_obs=True, sample_inside_tar=True):
        """
        simulate_one_trajectory: simulate the trajectory given the state or
            randomly initialized.

        Args:
            policy (func): agent's policy.
            T (int, optional): the maximum length of the trajectory. Defaults to 250.
            endType (str, optional): when to end the rollout. Defaults to 'TF'.
            state (np.ndarray, optional): if provided, set the initial state to
                its value. Defaults to None.
            theta (float, optional): if provided, set the theta to its value.
                Defaults to np.pi/2.
            sample_inside_obs (bool, optional): sampling initial states inside
                of the obstacles or not. Defaults to True.
            sample_inside_tar (bool, optional): sampling initial states inside
                of the targets or not. Defaults to True.

        Returns:
            np.ndarray: states of the trajectory, of the shape (length, 3).
            int: result.
            float: the minimum reach-avoid value of the trajectory.
            dictionary: extra information, (v_x, g_x, l_x, obs) along the trajectory.
        """
        # reset
        if state is None:
            _state = self.sample_state(sample_inside_obs, sample_inside_tar, theta=theta)
        else:
            _state = state
        result = 0 # not finished
        traj = []
        observations = []
        valueList = []
        gxList = []
        lxList = []

        for t in range(T):
            #= get obs, g, l
            obs = self._get_obs(_state)
            traj.append(_state)
            observations.append(obs)
            g_x = self.safety_margin(_state)
            l_x = self.target_margin(_state)

            #= add rollout record
            if t == 0:
                maxG = g_x
                current = max(l_x, maxG)
                minV = current
            else:
                maxG = max(maxG, g_x)
                current = max(l_x, maxG)
                minV = min(current, minV)

            valueList.append(minV)
            gxList.append(g_x)
            lxList.append(l_x)

            #= check the termination criterion
            if endType == 'end':
                done = not self.check_within_bounds(_state)
                if done:
                    result = -1
            elif endType == 'TF':
                if g_x > 0:
                    result = -1 # failed
                    break
                elif l_x <= 0:
                    result = 1  # succeeded
                    break
            elif endType == 'fail':
                if g_x > 0:
                    result = -1 # failed
                    break

            #= simulate
            action = policy(obs)
            w = self.getTurningRate(action)
            _state = self.integrate_forward(_state, w)

        traj = np.array(traj)
        observations = np.array(observations)
        info = {'valueList':valueList, 'gxList':gxList, 'lxList':lxList,
                'observations':observations}
        return traj, result, minV, info


    def simulate_trajectories(self, policy, num_rnd_traj=None, T=250, endType='TF',
        states=None, theta=np.pi/2, sample_inside_obs=True, sample_inside_tar=True):
        """
        simulate_trajectories: simulate the trajectories. If the states are not
            provided, we pick the initial states from the discretized state space.

        Args:
            policy (func): agent's policy.
            num_rnd_traj ([type], optional): [description]. Defaults to None.
            T (int, optional): the maximum length of the trajectory. Defaults to 250.
            endType (str, optional): when to end the rollout. Defaults to 'TF'.
            states (np.ndarray, optional): if provided, set the initial states to
                its value. Defaults to None.
            theta (float, optional): if provided, set the theta to its value.
                Defaults to np.pi/2.
            sample_inside_obs (bool, optional): sampling initial states inside
                of the obstacles or not. Defaults to True.
            sample_inside_tar (bool, optional): sampling initial states inside
                of the targets or not. Defaults to True.

        Returns:
            list of np.ndarray: each element is a tuple consisting of x and y
                positions along the trajectory.
            np.ndarray: the binary reach-avoid outcomes.
            np.ndarray: the minimum reach-avoid values of the trajectories.
        """
        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = []

        if states is None:
            results = np.empty(shape=(num_rnd_traj,), dtype=int)
            minVs = np.empty(shape=(num_rnd_traj,), dtype=float)
            for idx in range(num_rnd_traj):
                traj, result, minV, _ = self.simulate_one_trajectory(
                    policy, T=T, endType=endType, theta=theta,
                    sample_inside_obs=sample_inside_obs,
                    sample_inside_tar=sample_inside_tar)
                trajectories.append(traj)
                results[idx] = result
                minVs[idx] = minV
        else:
            results = np.empty(shape=(len(states),), dtype=int)
            minVs = np.empty(shape=(len(states),), dtype=float)
            for idx, state in enumerate(states):
                traj, result, minV, _ = self.simulate_one_trajectory(
                    policy, T=T, state=state, endType=endType)
                trajectories.append(traj)
                results[idx] = result
                minVs[idx] = minV

        return trajectories, results, minVs


    #== Plotting ==
    def visualize(  self, q_func, policy, device, rndTraj=False, num_rnd_traj=10,
                    vmin=-1, vmax=1, nx=51, ny=51, cmap='seismic',
                    labels=None, boolPlot=False, plotV=True, normalize_v=False):
        """
        visualize

        Args:
            q_func (object): agent's Q-network.
            policy (func): agent's policy.
            device (str): agent's device.
            rndTraj (bool, optional): random initialization or not. Defaults to False.
            num_rnd_traj (int, optional): number of states. Defaults to None.
            vmin (int, optional): vmin in colormap. Defaults to -1.
            vmax (int, optional): vmax in colormap. Defaults to 1.
            nx (int, optional): # points in x-axis. Defaults to 101.
            ny (int, optional): # points in y-axis. Defaults to 101.
            cmap (str, optional): color map. Defaults to 'seismic'.
            labels (list, optional): x- and y- labels. Defaults to None.
                v[idx] = q_func(obsTensor).max(dim=1)[0].cpu().detach().numpy()
            boolPlot (bool, optional): plot the binary values. Defaults to False.
        """
        thetaList = [np.pi, np.pi/2, 0]
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        axList = [ax1, ax2, ax3]

        for i, (ax, theta) in enumerate(zip(axList, thetaList)):
            if i == len(thetaList)-1:
                cbarPlot=True
            else:
                cbarPlot=False

            #== Plot failure / target set ==
            self.plot_target_failure_set(ax)

            #== Plot V ==
            if plotV:
                self.plot_v_values( q_func, device, fig, ax, theta=theta,
                                    boolPlot=boolPlot, cbarPlot=cbarPlot,
                                    vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
                                    normalize_v=normalize_v)

            #== Plot Trajectories ==
            thetas = theta*np.ones(shape=(self.visual_initial_states.shape[0], 1))
            states = np.concatenate((self.visual_initial_states, thetas), axis=1)
            if rndTraj:
                self.plot_trajectories( policy, ax, num_rnd_traj=num_rnd_traj,
                    theta=theta, endType='TF')
            else:
                self.plot_trajectories( policy, ax, states=states, endType='TF')

            #== Formatting ==
            self.plot_formatting(ax, labels=labels)
            fig.tight_layout()

            ax.set_xlabel(r'$\theta={:.0f}^\circ$'.format(theta*180/np.pi), fontsize=28)


    def plot_v_values(self, q_func, device, fig, ax, theta=np.pi/2,
            boolPlot=False, cbarPlot=True, vmin=-1, vmax=1, nx=101, ny=101,
            cmap='seismic', normalize_v=False):
        """
        plot_v_values

        Args:
            q_func (object): agent's Q-network.
            device (str): agent's device.
            fig (matplotlib.figure)
            ax (matplotlib.axes.Axes)
            theta (float, optional): if provided, fix the car's heading angle to
                its value. Defaults to np.pi/2.
            boolPlot (bool, optional): plot the values in binary form.
                Defaults to False.
            cbarPlot (bool, optional): plot the color bar or not. Defaults to True.
            vmin (int, optional): vmin in colormap. Defaults to -1.
            vmax (int, optional): vmax in colormap. Defaults to 1.
            nx (int, optional): # points in x-axis. Defaults to 101.
            ny (int, optional): # points in y-axis. Defaults to 101.
            cmap (str, optional): color map. Defaults to 'seismic'.
        """
        axStyle = self.get_axes()

        #== Plot V ==
        if theta == None:
            theta = 2.0 * np.random.uniform() * np.pi
        v, xs, ys = self.get_value(q_func, device, theta, nx, ny)

        if boolPlot:
            im = ax.imshow(v.T>0., interpolation='none', extent=axStyle[0],
                origin="lower", cmap=cmap, zorder=-1)
        else:
            if normalize_v:
                v = (v-np.min(v))/(np.max(v)-np.min(v))
                v = vmin + v*(vmax-vmin)
            im = ax.imshow(v.T, interpolation='none', extent=axStyle[0], origin="lower",
                    cmap=cmap, vmin=vmin, vmax=vmax, zorder=-1)
            CS = ax.contour(xs, ys, v.T, levels=[0], colors='g', linewidths=2,
                            linestyles='dashed')
            if cbarPlot:
                if normalize_v: # use true range in colorbar
                    vmin = np.min(v)
                    vmax = np.max(v)
                cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax])
                cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)


    def plot_trajectories(self, policy, ax, num_rnd_traj=None, T=250, endType='TF',
            states=None, theta=np.pi/2, sample_inside_obs=True, sample_inside_tar=True,
            c='k', lw=2, zorder=2):
        """
        plot_trajectories: plot trajectories given the agent's Q-network.

        Args:
            policy (func): agent's policy.
            ax (matplotlib.axes.Axes).
            num_rnd_traj (int, optional): Defaults to None.
            T (int, optional): the maximum length of the trajectory. Defaults to 250.
            endType (str, optional): when to end the rollout. Defaults to 'TF'.
            states (np.ndarray, optional): if provided, set the initial states to
                its value. Defaults to None.
            theta (float, optional): if provided, set the theta to its value.
                Defaults to np.pi/2.
            sample_inside_obs (bool, optional): sampling initial states inside
                of the obstacles or not. Defaults to True.
            sample_inside_tar (bool, optional): sampling initial states inside
                of the targets or not. Defaults to True.
            c (str, optional): color. Defaults to 'k'.
            lw (float, optional): linewidth. Defaults to 1.5.
            zorder (int, optional): graph layers order. Defaults to 2.

        Returns:
            np.ndarray: the binary reach-avoid outcomes.
            np.ndarray: the minimum reach-avoid values of the trajectories.
        """
        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))

        trajectories, results, minVs = self.simulate_trajectories(
            policy, num_rnd_traj=num_rnd_traj, T=T, endType=endType,
            states=states, theta=theta, sample_inside_obs=sample_inside_obs,
            sample_inside_tar=sample_inside_tar)

        if ax == None:
            ax = plt.gca()
        for traj in trajectories:
            traj_x = traj[:,0]
            traj_y = traj[:,1]
            ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
            ax.plot(traj_x, traj_y, color=c,  linewidth=lw, zorder=zorder)

        return results, minVs


    def plot_target_failure_set(self, ax, c_c='m', c_t='y', lw=3, zorder=0):
        """
        plot_target_failure_set

        Args:
            ax (matplotlib.axes.Axes).
            c_c (str, optional): the color of constraint set boundary. Defaults to 'm'.
            c_t (str, optional): the color of target set boundary. Defaults to 'y'.
            lw (int, optional): the linewidth of the boundaries. Defaults to 3.
            zorder (int, optional): the graph oder of the boundaries. Defaults to 0.
        """
        plot_circle(self._obs_loc, self._obs_radius, ax,
            c=c_c, lw=lw, zorder=zorder)
        plot_circle(self._goal_loc, self._goal_radius, ax, c=c_t,
            lw=lw, zorder=zorder)


    def plot_formatting(self, ax, labels=None, fsz=20):
        """
        plot_formatting

        Args:
            ax (matplotlib.axes.Axes).
            labels (list, optional): x- and y- labels. Defaults to None.
            fsz (int, optional): font size. Defaults to 20.
        """        
        axStyle = self.get_axes()
        # ax.plot([0., 0.], [axStyle[0][2], axStyle[0][3]], c='k')
        # ax.plot([axStyle[0][0], axStyle[0][1]], [0., 0.], c='k')
        #== Formatting ==
        ax.axis(axStyle[0])
        ax.set_aspect(axStyle[1])  # makes equal aspect ratio
        ax.grid(False)
        if labels is not None:
            ax.set_xlabel(labels[0], fontsize=fsz)
            ax.set_ylabel(labels[1], fontsize=fsz)

        ax.tick_params( axis='both', which='both',  # both x and y axes, both major and minor ticks are affected
                        bottom=False, top=False,    # ticks along the top and bottom edges are off
                        left=False, right=False)    # ticks along the left and right edges are off
        ax.xaxis.set_major_locator(LinearLocator(5))
        ax.xaxis.set_major_formatter('{x:.1f}')
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.yaxis.set_major_formatter('{x:.1f}')


    ################ Not used in episode ################
    def sample_tasks(self, num_tasks):
        # xpos = radius*np.cos(angle)
        # ypos = radius*np.sin(angle)
        # self.goals = np.vstack((xpos, ypos)).T
        goal = [1.0, 0.0]
        obs_loc = [0.5, 0.0]
        obs_radius = 0.1
        tasks = [{'goal': goal, 'obs_loc': obs_loc, 'obs_radius': obs_radius}
            for _ in range(num_tasks)]
        return tasks