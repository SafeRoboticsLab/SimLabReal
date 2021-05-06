import torch

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from maml_rl.metalearners.base import GradientBasedMetaLearner
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
									   to_numpy, vector_to_parameters)
from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.utils.reinforcement_learning import reinforce_loss, reinforce_latent_loss
from torch.distributions import MultivariateNormal
from collections import OrderedDict


class TRPO(GradientBasedMetaLearner):
	"""Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning
	application, with an outer-loop optimization based on TRPO [2].

	Parameters
	----------
	policy : `maml_rl.policies.Policy` instance
		The policy network to be optimized. Note that the policy network is an
		instance of `torch.nn.Module` that takes observations as input and
		returns a distribution (typically `Normal` or `Categorical`).

	fast_lr : float
		Step-size for the inner loop update/fast adaptation.

	num_steps : int
		Number of gradient steps for the fast adaptation. Currently setting
		`num_steps > 1` does not resample different trajectories after each
		gradient steps, and uses the trajectories sampled from the initial
		policy (before adaptation) to compute the loss at each step.

	first_order : bool
		If `True`, then the first order approximation of MAML is applied.

	device : str ("cpu" or "cuda")
		Name of the device for the optimization.

	References
	----------
	.. [1] Finn, C., Abbeel, P., and Levine, S. (2017). Model-Agnostic
		   Meta-Learning for Fast Adaptation of Deep Networks. International
		   Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

	.. [2] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., and Abbeel, P.
		   (2015). Trust Region Policy Optimization. International Conference on
		   Machine Learning (ICML) (https://arxiv.org/abs/1502.05477)
	"""
	def __init__(self,
				 policy,
				 fast_lr=0.5,
				 first_order=False,
				 latent_size=2,
				 device='cpu'):
		super(TRPO, self).__init__(policy, device=device)
		self.fast_lr = fast_lr
		self.first_order = first_order
		self.latent_size = latent_size

	async def adapt(self, train_futures, first_order=None):
		if first_order is None:
			first_order = self.first_order
		# Loop over the number of steps of adaptation
		# nn_params = None

		# if self.latent_size > 0:
		# 	mu = torch.zeros((self.latent_size), requires_grad=True)
		# 	logvar = torch.zeros((self.latent_size), requires_grad=True)
		# 	latent_params = OrderedDict(mu=mu, logvar=logvar)
		# 	latent_dist = MultivariateNormal(mu, torch.diag(torch.exp(logvar)))
		# else:
		# 	latent_params = None
		# 	latent_dist = None

		# adapt
		for futures in train_futures:
			train_episodes = await futures
			# if self.latent_size > 0:
			# 	num_step = train_episodes.observations.shape[0]
			# 	latent = train_episodes.latent.unsqueeze(0).repeat(num_step,1,1)
			# 	inner_loss = reinforce_latent_loss(train_episodes, 
			# 										latent=latent, 
			# 										latent_dist=latent_dist)
			# 	update_latent_params = True
			# 	update_nn_params = False
			# else:
			# 	inner_loss = reinforce_loss(self.policy, 
            #                     			train_episodes, 
			# 								params=nn_params)
			# 	update_latent_params = False
			# 	update_nn_params = True

			# inner_loss = reinforce_loss(self.policy,
			# 							train_episodes,
			# 							params=nn_params)
			# nn_params = self.policy.update_params(inner_loss,
											#    params=nn_params,
											#    step_size=self.fast_lr,
											#    first_order=first_order)
			# latent_params, nn_params = self.policy.update_params(inner_loss,
			# 						latent_params=latent_params,
			# 						nn_params=nn_params,
			# 						step_size=self.fast_lr,
			# 						first_order=first_order,
			# 						update_nn_params=update_nn_params,
			# 						update_latent_params=update_latent_params)
		# return latent_params, nn_params
		return 1

	def hessian_vector_product(self, kl, damping=1e-2):
		grads = torch.autograd.grad(kl,
									self.policy.parameters(),
									create_graph=True)
		flat_grad_kl = parameters_to_vector(grads)

		def _product(vector, retain_graph=True):
			grad_kl_v = torch.dot(flat_grad_kl, vector)
			grad2s = torch.autograd.grad(grad_kl_v,
										 self.policy.parameters(),
										 retain_graph=retain_graph)
			flat_grad2_kl = parameters_to_vector(grad2s)

			return flat_grad2_kl + damping * vector
		return _product

	async def surrogate_loss(self, train_futures, valid_futures, old_pi=None):
		first_order = (old_pi is not None) or self.first_order
		await self.adapt(train_futures, first_order=first_order)

		with torch.set_grad_enabled(old_pi is None):
			valid_episodes = await valid_futures
			# print(valid_episodes.observations.shape)
			if self.latent_size > 0:
				num_step = valid_episodes.observations.shape[0]
				latent = valid_episodes.latent.unsqueeze(0).repeat(num_step,1,1)
			else:
				latent = None
			pi = self.policy(valid_episodes.observations,
								# params=nn_params, 
								latent=latent)	#* no updated params for valid

			if old_pi is None:
				old_pi = detach_distribution(pi)

			log_ratio = (pi.log_prob(valid_episodes.actions)
						 - old_pi.log_prob(valid_episodes.actions))
			ratio = torch.exp(log_ratio)

			losses = -weighted_mean(ratio * valid_episodes.advantages,
									lengths=valid_episodes.lengths)
			kls = weighted_mean(kl_divergence(pi, old_pi),
								lengths=valid_episodes.lengths)
		return losses.mean(), kls.mean(), old_pi

	def step(self,
			 train_futures,
			 valid_futures,
			 max_kl=1e-3,
			 cg_iters=10,
			 cg_damping=1e-2,
			 ls_max_steps=10,
			 ls_backtrack_ratio=0.5):
		# num_tasks = len(train_futures[0])
		# num_tasks = len(valid_futures[0])
		num_tasks = 20

		# logs = {}
		# Compute the surrogate loss (inner loop update)
		old_losses, old_kls, old_pis = self._async_gather([
			self.surrogate_loss(train, valid, old_pi=None)
			for (train, valid) in zip(zip(*train_futures), valid_futures)])
		# logs['loss_before'] = to_numpy(old_losses)
		# logs['kl_before'] = to_numpy(old_kls)

		old_loss = sum(old_losses) / num_tasks
		grads = torch.autograd.grad(old_loss,
									self.policy.parameters(),
									retain_graph=True)
		grads = parameters_to_vector(grads)

		# Compute the step direction with Conjugate Gradient
		old_kl = sum(old_kls) / num_tasks
		hessian_vector_product = self.hessian_vector_product(old_kl,
															 damping=cg_damping)
		stepdir = conjugate_gradient(hessian_vector_product,
									 grads,
									 cg_iters=cg_iters)

		# Compute the Lagrange multiplier
		shs = 0.5 * torch.dot(stepdir,
							  hessian_vector_product(stepdir, retain_graph=False))
		lagrange_multiplier = torch.sqrt(shs / max_kl)

		step = stepdir / lagrange_multiplier

		# Save the old parameters
		old_params = parameters_to_vector(self.policy.parameters())

		# Line search
		step_size = 1.0
		for _ in range(ls_max_steps):
			vector_to_parameters(old_params - step_size * step,
								 self.policy.parameters())

			losses, kls, _ = self._async_gather([
				self.surrogate_loss(train, valid, old_pi=old_pi)
				for (train, valid, old_pi)
				in zip(zip(*train_futures), valid_futures, old_pis)])

			improve = (sum(losses) / num_tasks) - old_loss
			kl = sum(kls) / num_tasks
			if (improve.item() < 0.0) and (kl.item() < max_kl):
				# logs['loss_after'] = to_numpy(losses)
				# logs['kl_after'] = to_numpy(kls)
				break
			step_size *= ls_backtrack_ratio
		else:
			vector_to_parameters(old_params, self.policy.parameters())

		return None
