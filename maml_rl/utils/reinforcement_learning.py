import numpy as np
from maml_rl.utils.torch_utils import weighted_mean, to_numpy

def value_iteration(transitions, rewards, gamma=0.95, theta=1e-5):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    delta = np.inf
    while delta >= theta:
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        new_values = np.max(q_values, axis=1)
        delta = np.max(np.abs(new_values - values))
        values = new_values

    return values

def value_iteration_finite_horizon(transitions, rewards, horizon=10, gamma=0.95):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    for k in range(horizon):
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        values = np.max(q_values, axis=1)

    return values

def get_returns(episodes):
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])

#* Assume normalized, use the step with largest reward as the cost for the trajectory. Use positive value
def get_costs(episodes, scale=1):
    return to_numpy([-episode.rewards.max(dim=0)[0]*scale for episode in episodes])

#############################################################################3

def reinforce_loss(policy, episodes, params=None):
    
    #* re-evaluate the policy so gradient available
    pi = policy(episodes.observations.view((-1, *episodes.observation_shape)),
                params=params)
    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
    log_probs = log_probs.view(len(episodes), episodes.batch_size)  #! assuming action_shape == 1?
    losses = -weighted_mean(log_probs * episodes.advantages,
                            lengths=episodes.lengths)
    return losses.mean()

#* For inner update
def reinforce_latent_loss(episodes, latent, latent_dist):

	# log prob of gaussian
    log_probs = latent_dist.log_prob(latent)

    # print(log_probs.shape, episodes.advantages.shape)
    losses = -weighted_mean(log_probs * episodes.advantages,
                            lengths=episodes.lengths)
    return losses.mean()
