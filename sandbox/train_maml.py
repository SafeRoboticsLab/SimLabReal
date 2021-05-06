import gym
import torch
import json
import os
import yaml
from tqdm import trange
import numpy as np
from numpy import array
import random

import visdom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns, get_costs


def main(args):
	with open(args.config, 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	# Make output folder and save config
	if args.output_folder is not None:
		if not os.path.exists(args.output_folder):
			os.makedirs(args.output_folder)
		with open(os.path.join(args.output_folder, 'config.json'), 'w') as f:
			config.update(vars(args))
			json.dump(config, f, indent=2)

	# Set seed
	if args.seed is not None:
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)

	# Initialize gym env
	env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
	env.close()

	# Set sparse reward if specified
	env.sparse_reward = config['sparse_reward']

	# Policy
	policy = get_policy_for_env(env,
								latent_size=config['latent_size'],
								hidden_sizes=config['hidden-sizes'],
								nonlinearity=config['nonlinearity'])
	policy.share_memory()

	# Baseline
	baseline = LinearFeatureBaseline(get_input_size(env))

	# Sampler
	sampler = MultiTaskSampler(config['env-name'],
							   env_kwargs=config.get('env-kwargs', {}),
							   batch_size=config['fast-batch-size'],
							   latent_size=config['latent_size'],
							   policy=policy,
							   baseline=baseline,
							   env=env,
							   seed=args.seed,
							   num_workers=args.num_workers)

	# Meta learner
	metalearner = MAMLTRPO(policy,
						   fast_lr=config['fast-lr'],
						   first_order=config['first-order'],
						   latent_size=config['latent_size'],
						   device=args.device)

	# Visualize training progress
	if config['vis']:
		vis = visdom.Visdom(env=args.output_folder.split('/')[1])
		train_reward_window = vis.line(
			X=array([[0,]]), 
   			Y=array([[0,]]),
			opts=dict(xlabel='epoch',title='Avg train reward'))
		train_cost_window = vis.line(
			X=array([[0,]]), 
   			Y=array([[0,]]),
			opts=dict(xlabel='epoch',title='Avg train cost'))
		test_cost_window = vis.line(
			X=array([[0,]]), 
   			Y=array([[0,]]),
			opts=dict(xlabel='epoch',title='Avg test cost'))

	# Sample train and test tasks
	train_tasks_all = sampler.sample_tasks(num_tasks=100, uniform=False)	# should not uniform sampling since we do not know the distribution for PAC-Bayes
	# train_tasks = sampler.sample_tasks(num_tasks=config['num-train-task'], uniform=False)	# should not uniform sampling since we do not know the distribution for PAC-Bayes
	# print(train_tasks[0])

	# Train and test
	num_iterations = 0
	train_reward_all = []
	train_cost_all = []
	test_cost_all = []
	prev_policy_name = None
	for batch in trange(config['num-batches']):

		task_ind_chosen = np.random.choice(100, size=config['num-train-task'], replace=False).astype('int')
		train_tasks = [train_tasks_all[ind] for ind in task_ind_chosen]
		# train_tasks = sampler.sample_tasks(num_tasks=config['num-train-task'], uniform=False)	# should not uniform sampling since we do not know the distribution for PAC-Bayes

  		# Sample while updating inner, keeping a separate copy of theta
		futures = sampler.sample_async(train_tasks,
									   num_steps=config['num-steps'],
									   fast_lr=config['fast-lr'],
									   gamma=config['gamma'],
									   gae_lambda=config['gae-lambda'],
									   device=args.device)

		# Meta updating theta
		_ = metalearner.step(*futures,
								max_kl=config['max-kl'],
								cg_iters=config['cg-iters'],
								cg_damping=config['cg-damping'],
								ls_max_steps=config['ls-max-steps'],
								ls_backtrack_ratio=config['ls-backtrack-ratio'])

		#sys.exit()
		_, valid_episodes = sampler.sample_wait(futures)
		# num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
		# num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
		# logs.update(tasks=train_tasks,
		# 			num_iterations=num_iterations,
		# 			train_returns=get_returns(train_episodes[0]),
		# 			valid_returns=get_returns(valid_episodes))

		# Get reward
		mean_train_reward = np.mean(get_returns(valid_episodes))

		# Get cost, for each task use averaged cost over valid trajs, valid_returns.shape=num_tasks x num_valid_traj
		mean_train_cost = np.mean(np.mean(get_costs(valid_episodes), axis=1))

		# Debug
		print('Reward: %.4f, Cost: %.4f' % (mean_train_reward, mean_train_cost))

		# plot training progress
		if config['vis']:
			vis.line(X=array([[batch, ]]),
					Y=array([[mean_train_reward,]]),
					win=train_reward_window,update='append')
			vis.line(X=array([[batch, ]]),
					Y=array([[mean_train_cost,]]),
					win=train_cost_window,update='append')

		# Plot exploration, valid_eposodes: num_task x traj_length x num_traj x state_dim
		if batch % 5 == 0 and batch > 0:
			plt.figure()
			plt.plot(0., 0., 'ko', markersize=5, label='origin')	# starting pos

			# For navigation
			# for task in train_tasks:	# plot goals
				# plt.plot(task['goal'][0], task['goal'][1], 'ro', markersize=5)
			# for episode in valid_episodes:	# plot all 
				# # plt.plot(episode.observations[:,0:2,0], episode.observations[:,0:2,1], '-', lw=0.5)

			# For pusher, plot one task only
			task_to_plot_ind = random.randint(0, config['num-train-task']-1)
			task = train_tasks[task_to_plot_ind]
			# for task in train_tasks:	# plot goals
			for obj_ind in range(5):
				if obj_ind == task['goal'][0]:	# obj to be pushed
					plt.plot(task['goal'][3+2*obj_ind], task['goal'][4+2*obj_ind], 'co', markersize=5)
				else:
					plt.plot(task['goal'][3+2*obj_ind], task['goal'][4+2*obj_ind], 'go', markersize=5)
				plt.plot(task['goal'][1], task['goal'][2], 'ro', markersize=5)	# goal
			# for episode in valid_episodes:	# plot all
			episode = valid_episodes[task_to_plot_ind]
			# print(episode.observations.shape)	# step(100) x num_traj(50) x dim
			plt.plot(episode.observations[:,0:5,0], episode.observations[:,0:5,1], '-', lw=0.5)
			# plt.show()
			plt.xlim(-0.1, 1.0)
			plt.ylim(-1.0, 1.0)
			# plt.legend(loc='upper right')
			plt.savefig(args.output_folder+str(batch)+'_exp.png')
			plt.close()

		# Save policy if lowest training cost, delete old one
		if args.output_folder is not None and batch > 0 and mean_train_cost < min(train_cost_all):
			policy_name = args.output_folder+'step_'+str(batch)+'_cost_'+str(int(mean_train_cost*100))+'.th'
			torch.save(policy.state_dict(), policy_name)
			if prev_policy_name is not None:
				os.remove(prev_policy_name)
			prev_policy_name = policy_name

		# Test every 5 epochs
		if batch % 5 == 0 and batch > 0:
			test_tasks = sampler.sample_tasks(num_tasks=config['num-test-task'])

			_, test_valid_episodes = sampler.sample(test_tasks,
												num_steps=config['num-steps'],
												fast_lr=config['fast-lr'],
												gamma=config['gamma'],
												gae_lambda=config['gae-lambda'],
												device=args.device)
			mean_test_cost = np.mean(np.mean(get_costs(test_valid_episodes), axis=1))
			test_cost_all += [mean_test_cost]

			if config['vis']:
				vis.line(X=array([[batch, ]]),
						Y=array([[mean_test_cost,]]),
						win=test_cost_window,update='append')

		# Record
		train_reward_all += [mean_train_reward]
		train_cost_all += [mean_train_cost]
		torch.save({
			'epoch': batch,
			'train_reward_all': train_reward_all,
			'train_cost_all': train_cost_all,
			'test_cost_all': test_cost_all,
			'train_valid_episodes': valid_episodes,
			'train_tasks': train_tasks
		}, args.output_folder+'train_details')


if __name__ == '__main__':
	import argparse
	import multiprocessing as mp

	parser = argparse.ArgumentParser(description='Reinforcement learning with '
		'Model-Agnostic Meta-Learning (MAML) - Train')

	parser.add_argument('--config', type=str, required=True,
		help='path to the configuration file.')

	# Miscellaneous
	misc = parser.add_argument_group('Miscellaneous')
	misc.add_argument('--output-folder', type=str,
		help='name of the output folder')
	misc.add_argument('--seed', type=int, default=None,
		help='random seed')
	misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
		help='number of workers for trajectories sampling (default: '
			 '{0})'.format(mp.cpu_count() - 1))
	misc.add_argument('--use-cuda', action='store_true',
		help='use cuda (default: false, use cpu). WARNING: Full support for cuda is not guaranteed. Using CPU is encouraged.')

	args = parser.parse_args()
	# args.device = ('cuda' if (torch.cuda.is_available()
				#    and args.use_cuda) else 'cpu')
	args.device = 'cpu'
	main(args)
