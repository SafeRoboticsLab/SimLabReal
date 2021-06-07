import numpy as np
import random
import pybullet as pb
from pybullet_utils import bullet_client as bc
import gym

import os
os.sys.path.append(os.path.join(os.getcwd(), '.'))
from safe_rl.util_geom import euler2rot

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
    def __init__(self, task={},
                        img_H=96,
                        img_W=96,
                        render=True,
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

        # Set up observation and action space for Gym
        self.img_H = img_H
        self.img_W = img_W
        self.observation_space = gym.spaces.Box(low=0., high=1.,
            shape=(img_H, img_W), dtype=np.float32)

        # Car initial x/y/theta
        self.car_init_state = np.array([0.1, 0., 0.])

        # Car dynamics
        self.state_dim = 3
        self.action_dim = 1
        # TODO: tuning?
        self.v = 0.2
        # self.w = [-1.0, 0, 1.0]
        self.dt = 0.1
        # Discrete action space
        # self.action_space = gym.spaces.Discrete(3)
        # Continuous action space
        self.action_lim = np.array([1.])
        self.action_space = gym.spaces.Box(-self.action_lim, self.action_lim)
        self.doneType = doneType

        # Extract task info
        # TODO: specify more
        self._task = task
        self._goal_loc = task.get('goal_loc', np.array([self.state_bound-0.1, 0.]))
        self._goal_radius = task.get('goal_radius', 0.05)
        self._obs_loc  = task.get('obs_loc', np.array([self.state_bound/2, 0]))
        self._obs_radius = task.get('obs_radius', 0.1)
        self._obs_buffer = 0.1 # no cost if outside buffer

        # Set up PyBullet parameters
        self._renders = render
        self._physics_client_id = -1

        # Fix seed
        self.seed(0)


    def seed(self, seed=None):
        self.seed_val = seed
        self.action_space.seed(self.seed_val)
        np.random.seed(self.seed_val)
        random.seed(self.seed_val)
        # self.np_random, seed = gym.utils.seeding.np_random(seed)
        # return [seed]


    def reset_task(self, task):
        self._task = task
        self._goal_loc = task['goal_loc']
        self._goal_radius = task['goal_radius']
        self._obs_loc = task['obs_loc']
        self._obs_radius = task['obs_radius']


    def reset(self, random_init=False, sample_inside_obs=False, sample_inside_tar=True):
        if random_init:
            self._state = self.sample_state(sample_inside_obs, sample_inside_tar)
        else:
            self._state = self.car_init_state

        # Start PyBullet session if first time
        print("----------- reset simulation ---------------")
        if self._physics_client_id < 0:
            print("------------ start pybullet ----------------")
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
            self.ground_id = p.createMultiBody(
                baseMass=0,	# FIXED
                baseCollisionShapeIndex=ground_collision_id,
                basePosition=[	self.state_bound/2,
                                0,
                                -self.wall_thickness/2])
            wall_collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[	self.wall_thickness/2,
                                self.state_bound/2,
                                self.wall_height/2])
            wall_visual_id = -1
            if self._renders:
                wall_visual_id = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[	self.wall_thickness/2,
                                    self.state_bound/2,
                                    self.wall_height/2])
            self.wall_back_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision_id,
                baseVisualShapeIndex=wall_visual_id,
                basePosition=[	-self.wall_thickness/2,
                                0,
                                self.wall_height/2])
            self.wall_left_id = p.createMultiBody(	# positive in y
                baseMass=0,
                baseCollisionShapeIndex=wall_collision_id,
                baseVisualShapeIndex=wall_visual_id,
                basePosition=[	self.state_bound/2,
                                self.state_bound/2+self.wall_thickness/2,
                                self.wall_height/2],
                baseOrientation=p.getQuaternionFromEuler([0,0,np.pi/2]))
            self.wall_right_id = p.createMultiBody(	# negative in y
                baseMass=0,
                baseCollisionShapeIndex=wall_collision_id,
                baseVisualShapeIndex=wall_visual_id,
                basePosition=[	self.state_bound/2,
                                -self.state_bound/2-self.wall_thickness/2,
                                self.wall_height/2],
                baseOrientation=p.getQuaternionFromEuler([0,0,np.pi/2]))
            self.wall_front_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision_id,
                baseVisualShapeIndex=wall_visual_id,
                basePosition=[	self.state_bound+self.wall_thickness/2,
                                0,
                                self.wall_height/2])
            wall_top_visual_id = p.createVisualShape(p.GEOM_BOX,
                halfExtents=[	self.state_bound/2,
                                self.state_bound/2,
                                self.wall_thickness/2],
                rgbaColor=[1,1,1,0.1])
            self.wall_top_id = p.createMultiBody(	# for blocking view
                baseMass=0,
                baseCollisionShapeIndex=ground_collision_id,
                baseVisualShapeIndex=wall_top_visual_id,
                basePosition=[	self.state_bound/2,
                                0,
                                self.wall_height+self.wall_thickness/2])

            # Obstacle
            obs_collision_id = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=self._obs_radius,
                height=self.wall_height)
            obs_visual_id = -1
            if self._renders:
                obs_visual_id = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=self._obs_radius,
                    length=self.wall_height)
            self.obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=obs_collision_id,
                baseVisualShapeIndex=obs_visual_id,
                basePosition=np.append(self._obs_loc, self.wall_height/2))

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
        # pos, orn = p.getBasePositionAndOrientation(self.car_id)
        # euler = p.getEulerFromQuaternion(orn)
        return self._get_obs()


    def sample_state(self, sample_inside_obs=False, sample_inside_tar=True):
        # random sample `theta`
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


    def _get_obs(self):
        """
        Depth image not normalized right now
        """
        # State
        x,y,yaw = self._state
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
        _, _, _, depth, _ = self._p.getCameraImage(self.img_H, self.img_W,
            view_matrix, projection_matrix,
            flags=self._p.ER_NO_SEGMENTATION_MASK)
        depth = np.reshape(depth, (self.img_H, self.img_W))
        depth = far*near/(far - (far - near)*depth)
        return depth


    def step(self, action):
        # Determine turning rate
        # w = self.w[action]
        w = action

        #= Dynamics
        self._state = self.integrate_forward(self._state, w)
        # x,y,theta = self._state
        # x_new = x + self.v*np.cos(theta)*self.dt
        # y_new = y + self.v*np.sin(theta)*self.dt
        # theta_new = theta + w*self.dt
        # self._state = [x_new, y_new, theta_new]

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
        g_x = self.safety_margin(self._state)
        fail = g_x > 0
        success = l_x <= 0

        #= `reward` signal
        # TODO: tuning?
        dist_to_goal_center = np.linalg.norm(self._state[:2] - self._goal_loc)
        reward_goal = -dist_to_goal_center

        dist_to_obs_center = np.linalg.norm(self._state[:2] - self._obs_loc)
        dist_to_obs_boundary = dist_to_obs_center-self._obs_radius
        if dist_to_obs_center < self._obs_radius:
            reward_obs = -1
        elif dist_to_obs_center < (self._obs_radius+self._obs_buffer):
            reward_obs = -dist_to_obs_boundary/self._obs_buffer
        else:
            reward_obs = 0
        reward = reward_goal + reward_obs

        #= `done` signal
        if self.doneType == 'end':
            done = not self.check_within_bounds(self.state)
        elif self.doneType == 'fail':
            done = fail
        elif self.doneType == 'TF':
            done = fail or success
        else:
            raise ValueError("invalid doneType")

        if done and self.doneType == 'fail':
            g_x = 1  # TODO Tuning

        return self._get_obs(), reward, done, {'task': self._task,
            'state': self._state, 'g_x': g_x, 'l_x': l_x}


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


    #== GETTER ==
    def check_within_bounds(self, state):
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


    def safety_margin(self, state):
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
        return safety_margin


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test single environment in GUI
    env = NavigationObsPBEnv(render=True)
    print("\n== Environment Information ==")
    print("- state dim: {:d}, action dim: {:d}".format(env.state_dim, env.action_dim))
    print("- state bound: {:.2f}, done type: {}".format(env.state_bound, env.doneType))
    print("- action space:", env.action_space)
    # fig = plt.figure()
    # plt.imshow(obs, cmap='Greys')
    # plt.show()
    # print(env.safety_margin(np.array([-1.5, .5])))
    # print(env.safety_margin(np.array([1.5, .5])))
    # print(env.safety_margin(np.array([.5, 1.5])))
    # print(env.safety_margin(np.array([.5, -1.5])))

    # Run 3 trials
    # for i in range(3):
    #     print('\n== {} =='.format(i))
    obs = env.reset(random_init=False)
    for t in range(100):
        # Apply random action
        # action = random.randint(0,2)
        action = env.action_space.sample()[0]
        obs, r, done, info = env.step(action)
        state = info['state']

        # Debug
        x, y, yaw = state
        l_x = info['l_x']
        g_x = info['g_x']
        # print('[{}] a: {:.2f}, x: {:.3f}, y: {:.3f}, yaw: {:.3f}, r: {:.3f}, d: {}'.format(
        #     t, action, x, y, yaw, r, done))
        print('[{}] x: {:.3f}, y: {:.3f}, l_x: {:.3f}, g_x: {:.3f}, d: {}'.format(
            t, x, y, l_x, g_x, done))
        plt.imshow(obs, cmap='Greys')
        # plt.imshow(np.flip(np.swapaxes(obs, 0, 1), 1), cmap='Greys',
        # origin='lower')
        plt.show(block=False)    # Default is a blocking call
        plt.pause(.25)
        plt.close()
        if done:
            break