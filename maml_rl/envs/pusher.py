import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import random


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
	def __init__(self, task={}):

		self._task = task
		self._goal = task.get('goal', np.array((0,0,0,
										  		0.5,-0.5,
												0.6,-0.3,
											 	0.5,0,
											  	0.6,0.3,
											   	0.5,0.5), dtype=np.float32))
		# self._action_scaling = None

		# To be specfied
		self.sparse = None
		self.obj_bid = []

		utils.EzPickle.__init__(self)
		mujoco_env.MujocoEnv.__init__(self, '/home/allen/PAC-metaRL/maml_rl/envs/pusher_env.xml', 5)	# frame_skip


	def sample_tasks(self, num_tasks, uniform=False):

		goals = []
		for _ in range(num_tasks):
			curr_goal = np.zeros((13))

			# Sample goal
			curr_goal[0] = random.randint(0, 4)	# which block to push
			curr_goal[1] = np.random.uniform(.75, .95)	# goal
			# curr_goal[2] = np.random.uniform(-.5, .5)

			for i in range(5):
				xpos = np.random.uniform(.35, .65)
				ypos = np.random.uniform(-.5 + 0.2*i, -.3 + 0.2*i)
				# curr_goal[3+2*i] = -0.2*(blocknum + 1) + xpos	#* orig offset
				curr_goal[3+2*i] = xpos
				curr_goal[4+2*i] = ypos

			#* Make goal close to block (in y)
			curr_goal[2] = curr_goal[int(4+2*curr_goal[0])]

			goals.append(curr_goal)
		goals = np.asarray(goals)

		tasks = [{'goal': goal} for goal in goals]
		return tasks


	def reset_task(self, task):
		self._task = task
		self._goal = task['goal']


	def reset_model(self):

		# Hand
		qpos = self.init_qpos
		qvel = np.zeros_like(qpos)
		self.set_state(qpos, qvel)

		# Object
		for ind, obj_bid in enumerate(self.obj_bid):
			self.model.body_pos[obj_bid,:2] = self._goal[(3+ind*2):(5+ind*2)]
			# self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)

		# Target
		self.model.site_pos[self.target_obj_sid, :2] = self._goal[1:3]
		self.sim.forward()
		return self._get_obs()


	def _get_obs(self):
		"""
		* MAESN paper mentions using that but did not actually according to their codes
		"""

		obj_pos = np.empty((0))
		for obj_bid in self.obj_bid:
			obj_pos = np.concatenate((obj_pos, self.data.body_xpos[obj_bid,:2]))

		return np.concatenate([
			self.sim.data.qpos.flat[:3],	# joints, x/y/yaw for hand, and x/y for objects and target, 3+2*5+2=15
			obj_pos,
			self.sim.data.qvel.flat,
			self._goal[1:3],
		]).astype(np.float32).flatten()


	def step(self, action):

		#* No way to fix this unless modifying mujoco-py
		if len(self.obj_bid) == 0:
			for ind in range(5):
				self.obj_bid += [self.sim.model.body_name2id("obj"+str(ind))]
			self.target_obj_sid = self.sim.model.site_name2id("target")

		self.do_simulation(action, self.frame_skip)
		next_obs = self._get_obs()

		blockchoice = self._goal[0]
		curr_block_xidx = int(3 + 2*blockchoice)
		curr_block_yidx = int(4 + 2*blockchoice)
		#TODO: Maybe need to change angle here, specify angle?
		curr_block_pos = np.array([next_obs[curr_block_xidx], 
                             		next_obs[curr_block_yidx]])
		goal_pos = self._goal[1:3]

		# dist_to_block = np.linalg.norm(curr_gripper_pos -  curr_block_pos)
		block_dist = np.linalg.norm(goal_pos - curr_block_pos)
		goal_dist = np.linalg.norm(goal_pos)

		if self.sparse and block_dist > 0.2:
			reward = -goal_dist
		else:
			reward = -block_dist	# MAESN uses a factor of 5 here

		# Threshold
		if block_dist < 0.05 and goal_pos[0] > 0:	# avoid before initialization
			done = True
		else:
			done = False
		return next_obs, reward, done, {'task': self._task}


	####################### Debugging ##########################

	def viewer_setup(self):
		camera_id = self.model.camera_name2id('fixed')
		self.viewer.cam.type = 2
		self.viewer.cam.fixedcamid = camera_id
		self.viewer.cam.distance = self.model.stat.extent * 0.35
		# Hide the overlay
		self.viewer._hide_overlay = True

	def render(self, mode='human'):
		if mode == 'rgb_array':
			self._get_viewer(mode).render()
			# window size used for old mujoco-py:
			width, height = 500, 500
			data = self._get_viewer(mode).read_pixels(width, height, depth=False)
			return data
		elif mode == 'human':
			self._get_viewer(mode).render()


if __name__ == "__main__":
	env = PusherEnv()
	env.reset()
	env.step(np.zeros(3))

	pos = env.model.site_pos[env.sim.model.site_name2id("arm")]
	pos[2] -= 0.1
	print(pos)
	env.render()
	env.viewer.add_marker(pos=pos)
	env.render()

	import IPython
	IPython.embed()
