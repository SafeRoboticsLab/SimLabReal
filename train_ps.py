import gym
import torch
import json
import os
import yaml
from tqdm import trange
import numpy as np
from numpy import array
import visdom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from collections import OrderedDict
from torch.distributions.kl import kl_divergence

from maml_rl.metalearners import MAMLTRPOPS
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns, get_costs


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


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

	# Load prior policy
	policy = get_policy_for_env(env,
								latent_size=config['latent_size'],
								hidden_sizes=config['hidden-sizes'],
								nonlinearity=config['nonlinearity'])
	with open(config['prior-model-path'], 'rb') as f:
		state_dict = torch.load(f, map_location=torch.device(args.device))
		policy.load_state_dict(state_dict)
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

	# Posterior
	mu = torch.zeros((config['latent_size']), requires_grad=True)	# to be updated
	logvar = torch.zeros((config['latent_size']), requires_grad=True)	# sigma=exp(logvar/2)
	latent_dist = MultivariateNormal(mu, torch.diag(torch.exp(logvar)))
	latent_params = OrderedDict(mu=mu, logvar=logvar)

	# Prior
	mu_pr = torch.zeros((config['latent_size']), requires_grad=False)
	logvar_pr = torch.zeros((config['latent_size']), requires_grad=False)
	latent_pr = MultivariateNormal(mu_pr, torch.diag(torch.exp(logvar_pr)))

	# Meta learner for posterior
	metalearner = MAMLTRPOPS(policy,
						   latent_pr=latent_pr,
						   meta_lr=config['meta-lr'],
						   fast_lr=config['fast-lr'],
						   first_order=config['first-order'],
						   latent_size=config['latent_size'],
						   kl_ratio=config['kl-ratio'],
						   device=args.device)

	# Meta optimizer
	optimizer = torch.optim.Adam([
					{'params': mu, 'lr': config['meta-lr']},
					{'params': logvar, 'lr': 0.1*config['meta-lr']}])

	# Visualize training progress
	if config['vis']:
		vis = visdom.Visdom(env=args.output_folder.split('/')[1])
		train_reward_window = vis.line(
			X=array([[0,]]), 
   			Y=array([[0,]]),
			opts=dict(xlabel='epoch',title='RL train reward'))
		train_cost_window = vis.line(
			X=array([[0,]]), 
   			Y=array([[0,]]),
			opts=dict(xlabel='epoch',title='Avg train cost'))
		test_cost_window = vis.line(
			X=array([[0,]]), 
   			Y=array([[1,]]),	# since no value for epoch 0
			opts=dict(xlabel='epoch',title='Avg test cost'))
		bound_window = vis.line(
			X=array([[0,]]), 
   			Y=array([[0,]]),
			opts=dict(xlabel='epoch',title='Bound'))

	# Sample train and test tasks, not uniform
	train_tasks_all = sampler.sample_tasks(num_tasks=config['num-train-task'])
	test_tasks = sampler.sample_tasks(num_tasks=config['num-test-task'])

	# Train and test
	# num_iterations = 0
	train_cost_all = []
	train_reward_all = []
	test_cost_all = []
	bound_all = []
	prev_policy_name = None
	policy_name = None
	for batch in trange(config['num-batches']):

		task_ind_chosen = np.random.choice(config['num-train-task'], 20, replace=False).astype('int')
		train_tasks = [train_tasks_all[ind] for ind in task_ind_chosen]

		# Sample while updating inner, keeping a separate copy of theta
		latent_params_detach = OrderedDict()
		latent_params_detach['mu'] = latent_params['mu'].clone().detach()
		latent_params_detach['logvar'] =latent_params['logvar'].clone().detach()
		futures = sampler.sample_async(train_tasks,
									   latent_params=latent_params_detach,
									   num_steps=config['num-steps'],
									   fast_lr=config['fast-lr'],
									   gamma=config['gamma'],
									   gae_lambda=config['gae-lambda'],
									   device=args.device)

		# Get meta loss
		loss = metalearner.step(*futures,
								latent_dist=latent_dist,
								latent_params=latent_params)

		# Meta update
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Update latent dist
		latent_dist = MultivariateNormal(mu, torch.diag(torch.exp(logvar)))

		# Update logs
		train_episodes, valid_episodes = sampler.sample_wait(futures)
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

		# Estimate bound
		kld = kl_divergence(latent_dist, latent_pr).detach().numpy()
		reg = np.sqrt((kld + np.log(2*np.sqrt(config['num-train-task'])/0.01))/(2*config['num-train-task']))
		bound = 1-mean_train_cost-reg

		# Debug
		print('Reward: %.4f, Cost: %.4f, Bound: %.4f' % (mean_train_reward, mean_train_cost, bound))
		print('mu: ', mu)
		print('sigma: ', torch.exp(logvar/2))

		# Plot training progress
		if config['vis']:
			vis.line(X=array([[batch, ]]),
					Y=array([[mean_train_reward,]]),
					win=train_reward_window,update='append')
			vis.line(X=array([[batch, ]]),
					Y=array([[mean_train_cost,]]),
					win=train_cost_window,update='append')
			vis.line(X=array([[batch, ]]),
					Y=array([[bound,]]),
					win=bound_window,update='append')

		# Plot exploration, valid_eposodes: num_task x traj_length x num_traj x state_dim
		# plt.figure()
		# plt.plot(0., 0., 'go', markersize=5, label='origin')	# plot origin
		# for task_ind, task in enumerate(train_tasks):	# plot goals
		# 	if task_ind < 5:
		# 		plt.plot(task['goal'][0], task['goal'][1], 'ro', markersize=5)
		# for task_ind, episode in enumerate(valid_episodes):	# plot all
		# 	if task_ind < 5:
		# 		plt.plot(episode.observations[:,:2,0], episode.observations[:,:2,1], 'c-', lw=0.5)
		# plt.xlim(-1.2, 1.2)
		# plt.ylim(-0.2, 1.2)
		# # plt.legend(loc='upper right')
		# plt.savefig(args.output_folder+str(batch)+'_exp.png')
		# plt.close()

		# Test every 5 epochs
		if batch % 5 == 0 and batch > 0:
			latent_params_detach = OrderedDict()
			latent_params_detach['mu'] = latent_params['mu'].clone().detach()
			latent_params_detach['logvar'] = latent_params['logvar'].clone().detach()
			_, valid_episodes = sampler.sample(test_tasks,
									   		latent_params=latent_params_detach,
											num_steps=config['num-steps'],
											fast_lr=config['fast-lr'],
											gamma=config['gamma'],
											gae_lambda=config['gae-lambda'],
											device=args.device)
			mean_test_cost = np.mean(np.mean(get_costs(valid_episodes), axis=1))
			test_cost_all += [mean_test_cost]
			if config['vis']:
				vis.line(X=array([[batch, ]]),
						Y=array([[mean_test_cost,]]),
						win=test_cost_window,update='append')


		# Save posterior and policy if best training reward, delete old one
		if args.output_folder is not None and batch > 0 and mean_train_cost < min(train_cost_all):
			policy_name = args.output_folder+'step_'+str(batch)+'_cost_'+str(int(mean_train_cost*10000))+'.th'
			torch.save(policy.state_dict(), policy_name)
			if prev_policy_name is not None:
				os.remove(prev_policy_name)
			prev_policy_name = policy_name

		# Record
		train_cost_all += [mean_train_cost]
		train_reward_all += [mean_train_reward]
		bound_all += [bound]
		torch.save({
			'latent_params': latent_params,
			'policy_name': policy_name,
			'train_reward_all': train_reward_all,
			'train_cost_all': train_cost_all,
			'test_cost_all': test_cost_all,
			'bound_all': bound_all,
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
