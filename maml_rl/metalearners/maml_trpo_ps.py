import torch

# from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from maml_rl.metalearners.base import GradientBasedMetaLearner
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
									   to_numpy, vector_to_parameters)
# from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.utils.reinforcement_learning import reinforce_latent_loss
from torch.distributions import MultivariateNormal
# from collections import OrderedDict


class MAMLTRPOPS(GradientBasedMetaLearner):
	"""
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
	"""
	def __init__(self,
				 policy,
				 latent_pr,
				 meta_lr=0.001,
				 fast_lr=0.5,
				 first_order=False,
				 latent_size=2,
				 kl_ratio=1e-3,
				 device='cpu'):
		super(MAMLTRPOPS, self).__init__(policy, device=device)
		self.latent_pr = latent_pr
		self.meta_lr = meta_lr
		self.fast_lr = fast_lr
		self.first_order = first_order
		self.latent_size = latent_size
		self.kl_ratio = kl_ratio




	async def adapt(self, train_futures, 
                 		latent_dist, latent_params, first_order=None):
		if first_order is None:
			first_order = self.first_order
		# Loop over the number of steps of adaptation
		nn_params = None

		# adapt
		for futures in train_futures:
			train_episodes = await futures
			# print(train_episodes.latent.shape, train_episodes.observations.shape)
			num_step = train_episodes.observations.shape[0]
			latent = train_episodes.latent.unsqueeze(0).repeat(num_step,1,1)
			inner_loss = reinforce_latent_loss(train_episodes, 
												latent=latent, 
												latent_dist=latent_dist)

			latent_params, nn_params = self.policy.update_params(inner_loss,
													latent_params=latent_params,
													nn_params=nn_params,
													step_size=self.fast_lr,
													first_order=first_order,
													update_nn_params=False,
													update_latent_params=True)
		return latent_params, nn_params

	# def hessian_vector_product(self, kl, latent_params, damping=1e-2):
	# 	grads = torch.autograd.grad(kl,
	# 								# self.policy.parameters(),
	# 								latent_params.value(),
	# 								create_graph=True)
	# 	flat_grad_kl = parameters_to_vector(grads)

	# 	def _product(vector, retain_graph=True):
	# 		grad_kl_v = torch.dot(flat_grad_kl, vector)
	# 		grad2s = torch.autograd.grad(grad_kl_v,
	# 									#  self.policy.parameters(),
	# 									 latent_params.value(),
	# 									 retain_graph=retain_graph)
	# 		flat_grad2_kl = parameters_to_vector(grad2s)

	# 		return flat_grad2_kl + damping * vector
	# 	return _product

	async def surrogate_loss(self, train_futures, valid_futures, 
                          			latent_dist, latent_params):
		latent_params, nn_params = await self.adapt(train_futures,
													latent_dist,
													latent_params,
								  					first_order=False)

		with torch.set_grad_enabled(True):
		# with torch.set_grad_enabled(old_pi is None):
			valid_episodes = await valid_futures
			# print(valid_episodes.observations.shape)
			num_step = valid_episodes.observations.shape[0]
			latent = valid_episodes.latent.unsqueeze(0).repeat(num_step,1,1)

			# pi = self.policy(valid_episodes.observations,
            #         		params=nn_params, 
            #           		latent=latent)

			#* Use post-update latent params
			latent_dist = MultivariateNormal(latent_params['mu'], torch.diag(torch.exp(latent_params['logvar'])))

			# loss
			log_probs = latent_dist.log_prob(latent)
			losses = -weighted_mean(log_probs * valid_episodes.advantages,
									lengths=valid_episodes.lengths)

		return losses.mean(), None, None


	def step(self,
			 train_futures,
			 valid_futures,
			 latent_dist=None,
			 latent_params=None,
			#  max_kl=1e-3,
			#  cg_iters=10,
			#  cg_damping=1e-2,
			#  ls_max_steps=10,
			#  ls_backtrack_ratio=0.5
    		):
		num_tasks = len(train_futures[0])
		# logs = {}

		# Compute the surrogate loss (inner loop update)
		loss, _, _ = self._async_gather([
			self.surrogate_loss(train, valid, latent_dist, latent_params)
			for (train, valid) in zip(zip(*train_futures), valid_futures)])
		loss = sum(loss) / num_tasks

		# Add KL
		loss += self.kl_ratio*kl_divergence(latent_dist, self.latent_pr)

		# logs['loss'] = to_numpy(loss)
		# grads = torch.autograd.grad(loss,
		# 							# self.policy.parameters(),
		# 							latent_params.values(),
		# 							create_graph=True)
		# 							# retain_graph=True)
		# # torch.autograd.set_detect_anomaly(True)
		# # print(grads)

		# updated_latent_params = OrderedDict()
		# for (name, param), grad in zip(latent_params.items(), grads):
		# 	updated_latent_params[name] = param - self.meta_lr * grad
		# updated_latent_dist = MultivariateNormal(updated_latent_params['mu'],
		# 				torch.diag(torch.exp(updated_latent_params['logvar'])))

		# Compute the step direction with Conjugate Gradient
		# old_kl = sum(old_kls) / num_tasks
		# hessian_vector_product = self.hessian_vector_product(old_kl,
		# 													 latent_params,
		# 													 damping=cg_damping)
		# stepdir = conjugate_gradient(hessian_vector_product,
		# 							 grads,
		# 							 cg_iters=cg_iters)

		# # Compute the Lagrange multiplier
		# shs = 0.5 * torch.dot(stepdir,
		# 					  hessian_vector_product(stepdir, retain_graph=False))
		# lagrange_multiplier = torch.sqrt(shs / max_kl)

		# step = stepdir / lagrange_multiplier

		# # old_params = parameters_to_vector(self.policy.parameters())
		# old_params = latent_params.values()
		# print(type(old_params), old_params)

		# # Line search
		# step_size = 1.0
		# for _ in range(ls_max_steps):
			
		# 	vector_to_parameters(old_params - step_size * step,
		# 						 self.policy.parameters())

		# 	losses, kls, _ = self._async_gather([
		# 		self.surrogate_loss(train, valid, 
        #                			latent_dist, latent_params, 
        #                   		old_pi=old_pi)
		# 		for (train, valid, old_pi)
		# 		in zip(zip(*train_futures), valid_futures, old_pis)])

		# 	improve = (sum(losses) / num_tasks) - old_loss
		# 	kl = sum(kls) / num_tasks
		# 	if (improve.item() < 0.0) and (kl.item() < max_kl):
		# 		logs['loss_after'] = to_numpy(losses)
		# 		logs['kl_after'] = to_numpy(kls)
		# 		break
		# 	step_size *= ls_backtrack_ratio
		# else:	# useless
		# 	vector_to_parameters(old_params, self.policy.parameters())

		# return logs, updated_latent_dist, updated_latent_params
		return loss
