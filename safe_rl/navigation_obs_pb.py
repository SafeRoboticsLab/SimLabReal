import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client as bc
import gym
from utils_geom import euler2rot 

class NavigationObsPBEnv(gym.Env):
	"""Simple 2D navigation with obstacle using PyBullet. No dynamics/collision simulation. If PyBullet is not fast enough, we can use a renderer instead.

	State: x,y,theta of the car
	Dynamics: Dubins car, fixed forward speed v, turning rate {w,0,-w}, thus discrete control

	Right now there is a single obstacle at the center of the map. Info from step() stores distance to obstacle boundary (g) and distance to goal (l)

	"""
	def __init__(self, task={},
						img_H=96,
						img_W=96,
              			render=True):
		super(NavigationObsPBEnv, self).__init__()

		# Define dimensions
		self.action_lim = 0.04
		self.state_dim = 2
		self.wall_height = 1
		self.wall_thickness = 0.05
		self.car_dim = [0.04, 0.02, 0.01]	# half dims, only for visualization
		self.camera_height = 0.2	# cannot be too low, otherwise bad depth

		# Set up observation and action space for Gym
		self.img_H = img_H
		self.img_W = img_W
		self.observation_space = gym.spaces.Box(low=0., high=1., 
                                    	shape=(img_H, img_W), dtype=np.float32)
		# self.action_space = gym.spaces.Box(low=-self.action_lim, high=self.action_lim, shape=(self.state_dim,), dtype=np.float32)
		self.action_space = gym.spaces.Discrete(3)

		# Car initial x/y/theta
		self.car_init_state = [0.1,0,0]

		# Car dynamics - TODO: tuning?
		self.v = 0.2
		self.w = [-1.0,0,1.0]
		self.dt = 0.1

		# Extract task info - TODO: specify more
		self._task = task
		self._goal = task.get('goal', [self.state_dim-0.1, 0])
		self._obs_loc = task.get('obs_loc',[self.state_dim/2, 0])
		self._obs_radius = task.get('obs_radius', 0.1)
		self._obs_buffer = 0.1 # no cost if outside buffer
		self._goal_thres = 0.02

		# Set up PyBullet parameters
		self._renders = render
		self._physics_client_id = -1

		# Fix seed
		self.seed()

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]

	def reset_task(self, task):
		self._task = task
		self._goal = task['goal']
		self._obs_loc = task['obs_loc']
		self._obs_radius = task['obs_radius']

	def reset(self):
		self._state = np.zeros(self.state_dim, dtype=np.float32)

		# Start PyBullet session if first time
		print("-----------reset simulation---------------")
		if self._physics_client_id < 0:
			print("-----------start pybullet---------------")
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
				p.resetDebugVisualizerCamera(3.0, 180, -89, [self.state_dim/2, 0, 0])

			# Load ground, walls
			ground_collision_id = p.createCollisionShape(p.GEOM_BOX, 
                                    halfExtents=[self.state_dim/2, 
                                                self.state_dim/2, 
                                                self.wall_thickness/2])
			self.ground_id = p.createMultiBody(
       								baseMass=0,	# FIXED
               						baseCollisionShapeIndex=ground_collision_id,
									basePosition=[self.state_dim/2, 
                    								0, 
                        						-self.wall_thickness/2],
                 					)
			wall_collision_id = p.createCollisionShape(p.GEOM_BOX, 
                                    halfExtents=[self.wall_thickness/2, 
                                                self.state_dim/2, 
                                                self.wall_height/2])
			wall_visual_id = -1
			if self._renders:
				wall_visual_id = p.createVisualShape(p.GEOM_BOX, 
                                    halfExtents=[self.wall_thickness/2, 
                                                self.state_dim/2, 
                                                self.wall_height/2])
			self.wall_back_id = p.createMultiBody(
						baseMass=0,
						baseCollisionShapeIndex=wall_collision_id,
						baseVisualShapeIndex=wall_visual_id,
						basePosition=[-self.wall_thickness/2, 0, self.wall_height/2]
						)
			self.wall_left_id = p.createMultiBody(	# positive in y
						baseMass=0,
						baseCollisionShapeIndex=wall_collision_id,
						baseVisualShapeIndex=wall_visual_id,
						basePosition=[self.state_dim/2, 
                    					self.state_dim/2+self.wall_thickness/2, 
                        				self.wall_height/2],
						baseOrientation=p.getQuaternionFromEuler([0,0,np.pi/2])
						)
			self.wall_right_id = p.createMultiBody(	# negative in y
						baseMass=0,
						baseCollisionShapeIndex=wall_collision_id,
						baseVisualShapeIndex=wall_visual_id,
						basePosition=[self.state_dim/2, 
                    				-self.state_dim/2-self.wall_thickness/2, 
                        				self.wall_height/2],
						baseOrientation=p.getQuaternionFromEuler([0,0,np.pi/2])
						)
			self.wall_front_id = p.createMultiBody(
						baseMass=0,
						baseCollisionShapeIndex=wall_collision_id,
						baseVisualShapeIndex=wall_visual_id,
						basePosition=[self.state_dim+self.wall_thickness/2, 0, self.wall_height/2]
						)
			wall_top_visual_id = p.createVisualShape(p.GEOM_BOX, 
                                    halfExtents=[self.state_dim/2, 
                                                self.state_dim/2, 
                                                self.wall_thickness/2],
                                    rgbaColor=[1,1,1,0.1])
			self.wall_top_id = p.createMultiBody(	# for blocking view
						baseMass=0,
						baseCollisionShapeIndex=ground_collision_id,
						baseVisualShapeIndex=wall_top_visual_id,
						basePosition=[self.state_dim/2, 
                    					0, 
                        			self.wall_height+self.wall_thickness/2])
   
			# Obstacle
			obs_collision_id = p.createCollisionShape(p.GEOM_CYLINDER,
							radius=self._obs_radius, height=self.wall_height)
			obs_visual_id = -1
			if self._renders:
				obs_visual_id = p.createVisualShape(p.GEOM_CYLINDER, 
                            radius=self._obs_radius, length=self.wall_height)
			self.obs_id = p.createMultiBody(
						baseMass=0,
						baseCollisionShapeIndex=obs_collision_id,
						baseVisualShapeIndex=obs_visual_id,
						basePosition=self._obs_loc+[self.wall_height/2]
						)

			# Set up car if visualizing in GUI
			if self._renders:
				car_collision_id = p.createCollisionShape(p.GEOM_BOX, 
												halfExtents=self.car_dim)
				car_visual_id = -1
				car_visual_id = p.createVisualShape(p.GEOM_BOX, 
                                            halfExtents=self.car_dim,
											rgbaColor=[0,0,1,1])
				self.car_id = p.createMultiBody(
						baseMass=0, # no dynamics
						baseCollisionShapeIndex=car_collision_id,
						baseVisualShapeIndex=car_visual_id,
						basePosition=self.car_init_state[:2]+[0.03],
						baseOrientation=p.getQuaternionFromEuler([0,0,0]))

		p.stepSimulation()
		# pos, orn = p.getBasePositionAndOrientation(self.car_id)
		# euler = p.getEulerFromQuaternion(orn)
		self._state = self.car_init_state
		return self._get_obs()


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
		projection_matrix = self._p.computeProjectionMatrixFOV(fov=120.0, aspect=aspect, nearVal=near, farVal=far)
		_, _, _, depth, _ = self._p.getCameraImage(self.img_H, self.img_W, view_matrix, projection_matrix, flags=self._p.ER_NO_SEGMENTATION_MASK)
		depth = np.reshape(depth, (self.img_H, self.img_W))
		depth = far*near/(far - (far - near)*depth)
		return depth


	def step(self, action):
		# Determine turning rate
		w = self.w[action]

		# Dynamics
		x,y,theta = self._state
		x_new = x + self.v*np.cos(theta)*self.dt
		y_new = y + self.v*np.sin(theta)*self.dt
		theta_new = theta + w*self.dt
		self._state = [x_new, y_new, theta_new]

		# Move car in simulation - not necessary if not visualizing in GUI - since all we need from PyBullet is simulation of the camera and the obstacle field
		if self._renders:
			self._p.resetBasePositionAndOrientation(self.car_id,
								[x_new,y_new,0.03],
								self._p.getQuaternionFromEuler([0,0,theta_new]))

		# Reward: goal and obstacle  TODO: tuning?
		x_goal = x - self._goal[0]
		y_goal = y - self._goal[1]
		x_obs = x - self._obs_loc[0]
		y_obs = y - self._obs_loc[1]
		l = np.sqrt(x_goal ** 2 + y_goal ** 2)
		reward_goal = -l

		dist_to_back_wall = x
		dist_to_left_wall = self.state_dim/2-y
		dist_to_right_wall = self.state_dim/2+y
		dist_to_front_wall = self.state_dim - x
		dist_to_obs_center = np.sqrt(x_obs ** 2 + y_obs ** 2)
		dist_to_obs_boundary = dist_to_obs_center-self._obs_radius
		g = min([dist_to_back_wall, dist_to_left_wall, dist_to_right_wall, dist_to_front_wall, dist_to_obs_boundary])
		# print(dist_to_obs_center, self._goal, self._obs_loc, self._obs_radius)
		if dist_to_obs_center < self._obs_radius:
			reward_obs = -1
		elif dist_to_obs_center < (self._obs_radius+self._obs_buffer):
			reward_obs = -dist_to_obs_boundary/self._obs_buffer
		else:
			reward_obs = 0
		reward = reward_goal + reward_obs
		done = (l < self._goal_thres)
		return self._get_obs(), reward, done, {'task': self._task, 
											'state': self._state,
                                     		'g': g,
                                       		'l': l}

	################ Not used in episode ################

	def sample_tasks(self, num_tasks):
		# xpos = radius*np.cos(angle)
		# ypos = radius*np.sin(angle)
		# self.goals = np.vstack((xpos, ypos)).T
		goal = [1.0, 0.0]	
		obs_loc = [0.5, 0.0]
		obs_radius = 0.1	
		tasks = [{'goal': goal, 'obs_loc': obs_loc, 'obs_radius': obs_radius} for _ in range(num_tasks)]
		return tasks


if __name__ == '__main__':	
	import matplotlib.pyplot as plt
	import random
 
	# Test single environment in GUI
	env = NavigationObsPBEnv()
	obs = env.reset()
	# fig = plt.figure()
	# plt.imshow(obs, cmap='Greys')
	# plt.show()

	# Run one trial
	for t in range(100):

		# Apply random action
		action = random.randint(0,2)
		obs, reward, done, info = env.step(action)
		state = info['state']

		# Debug
		print(f'x: {state[0]}, y: {state[1]}, yaw: {state[2]}, reward: {reward}, done: {done}')
		plt.imshow(obs, cmap='Greys')
		# plt.imshow(np.flip(np.swapaxes(obs, 0, 1), 1), cmap='Greys', origin='lower')
		plt.show()    # Default is a blocking call
