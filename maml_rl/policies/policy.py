import torch
import torch.nn as nn
import sys
from collections import OrderedDict

def weight_init(module):
	if isinstance(module, nn.Linear):
		nn.init.xavier_uniform_(module.weight)
		module.bias.data.zero_()

class Policy(nn.Module):
	def __init__(self, input_size, output_size):
		super(Policy, self).__init__()
		self.input_size = input_size
		self.output_size = output_size

		# For compatibility with Torchmeta
		self.named_meta_parameters = self.named_parameters
		self.meta_parameters = self.parameters

	def update_params(self, loss, latent_params=None, nn_params=None, step_size=0.5, first_order=False, update_nn_params=False, update_latent_params=False):
		"""Apply one step of gradient descent on the loss function `loss`, with 
		step-size `step_size`, and returns the updated parameters of the neural 
		network.
		"""

		# When adapting online, first copy over nn parameters
		if nn_params is None:
			nn_params = OrderedDict(self.named_meta_parameters())

		# Update latent parameters if specified
		if update_latent_params:
			latent_grads = torch.autograd.grad(loss, latent_params.values(),
										create_graph=not first_order,
										allow_unused=True)
			updated_latent_params = OrderedDict()
			for (name, param), grad in zip(latent_params.items(), latent_grads):
				updated_latent_params[name] = param - step_size * grad
		else:
			updated_latent_params = latent_params

		# Update nn parameters if specified
		if update_nn_params:
			nn_grads = torch.autograd.grad(loss, nn_params.values(),
										create_graph=not first_order,
										allow_unused=True)
			updated_nn_params = OrderedDict()
			for (name, param), grad in zip(nn_params.items(), nn_grads):
				updated_nn_params[name] = param - step_size * grad
		else:
			updated_nn_params = nn_params

		return updated_latent_params, updated_nn_params
