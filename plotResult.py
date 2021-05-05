import numpy as np
from numpy import array
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import torch


def main(trial_name):

	training_details_dic_path = 'models/'+trial_name+'/train_details'
	training_details_dic = torch.load(training_details_dic_path)

	train_valid_episodes = training_details_dic['train_valid_episodes']

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(0., 0., 'ko', markersize=5, label='origin')	# starting pos

	# task_to_plot_ind = random.randint(0, config['num-train-task']-1)
	# task = train_tasks[task_to_plot_ind]
	# for task in train_tasks:	# plot goals
	# for obj_ind in range(5):
	# 	if obj_ind == task['goal'][0]:	# obj to be pushed
	# 		plt.plot(task['goal'][3+2*obj_ind], task['goal'][4+2*obj_ind], 'co', markersize=5)
	# 	else:
	# 		plt.plot(task['goal'][3+2*obj_ind], task['goal'][4+2*obj_ind], 'go', markersize=5)
	# 	plt.plot(task['goal'][1], task['goal'][2], 'ro', markersize=5)	# goal
	for episode in train_valid_episodes:	# plot all
	# episode = valid_episodes[task_to_plot_ind]
	# print(episode.observations.shape)	# step(100) x num_traj(50) x dim
		plt.plot(episode.observations[:,0:5,0], episode.observations[:,0:5,1], '-', lw=1.0)
	rect1 = matplotlib.patches.Rectangle((0.35,-0.5), 0.3, 1.0, color='orange')
	rect2 = matplotlib.patches.Rectangle((0.75,-0.5), 0.2, 1.0, color='cyan')
	ax.add_patch(rect1)
	ax.add_patch(rect2)
	plt.xlim(0.0, 0.9)
	plt.ylim(-0.5, 0.5)
	# plt.legend(loc='upper right')
	plt.show()

	return

	train_reward_all = training_details_dic['train_reward_all']
	train_cost_all = training_details_dic['train_cost_all']
	test_cost_all = training_details_dic['test_cost_all']
	bound_all = training_details_dic['bound_all']
	num_step = len(train_reward_all)
	num_step_test = len(test_cost_all)
	# 'latent_params': latent_params,
	# 'policy_name': policy_name,

	# Plot pose in 2D
	# plt.imshow(depth, cmap='Greys', interpolation='nearest') 
	# plt.scatter(x=pixel_ind_all[1], y=pixel_ind_all[0], s=15, c='red')  # pixels use coordinates from left_top corner, and height as x; scatter uses coordinates from left_bottom corner, and normal x
	# plt.scatter(x=pixel_ind_all_new[1], y=pixel_ind_all_new[0], s=15, c='green')
	# plt.figure()

	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.plot(np.arange(num_step), train_reward_all, color='grey', label='emp')
	ax1.plot(np.arange(num_step), train_cost_all, color='red', label='bound')
	ax1.legend()

	ax2.plot(np.arange(num_step_test)*5, test_cost_all, color='grey', label='cost env')
	ax2.plot(np.arange(num_step), bound_all, color='red', label='reg cost')
	ax2.legend()

	plt.show()


if __name__ == '__main__':

	import argparse
	def collect_as(coll_type):
		class Collect_as(argparse.Action):
			def __call__(self, parser, namespace, values, options_string=None):
				setattr(namespace, self.dest, coll_type(values))
		return Collect_as
	parser = argparse.ArgumentParser(description='PAC-Bayes Opt')
	parser.add_argument('--trial_name', type=str)
	arg_con = parser.parse_args()
	trial_name = arg_con.trial_name  # 'grasp_pac_1/'

	main(trial_name=trial_name)
