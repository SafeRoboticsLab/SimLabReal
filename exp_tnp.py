#!/usr/bin/env python
# coding: utf-8

# In[1]:


from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import types
import pickle
import time
timestr = time.strftime("%Y-%m-%d-%H_%M")

#= AGENT
from RARL.utils import save_obj
from RARL.DDQN_image import DDQN_image
from RARL.config import dqnConfig

#= ENV
from safe_rl.navigation_obs_pb_disc import NavigationObsPBEnvDisc


# In[2]:


# Test single environment
env_name = 'navigation_pac_ra_disc-v0'
render=False
doneType='end'
img_sz = 48
env = NavigationObsPBEnvDisc(render=render, img_H=img_sz, img_W=img_sz, doneType=doneType)

stateDim = env.state_dim
actionNum = env.action_num
actionSet = env.discrete_controls
env.report()

fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
env.plot_target_failure_set(ax=ax)
env.plot_formatting(ax=ax, labels=['x', 'y'], fsz=16)
plt.show()


# In[3]:


nx, ny = 101, 101
vmin = -1
vmax = 1

v = np.zeros((nx, ny))
l_x = np.zeros((nx, ny))
g_x = np.zeros((nx, ny))
xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
ys =np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

it = np.nditer(v, flags=['multi_index'])

while not it.finished:
    idx = it.multi_index
    x = xs[idx[0]]
    y = ys[idx[1]]

    l_x[idx] = env.target_margin(np.array([x, y]))
    g_x[idx] = env.safety_margin(np.array([x, y]))

    v[idx] = np.maximum(l_x[idx], g_x[idx])
    it.iternext()

axStyle = env.get_axes()

fig, axes = plt.subplots(1,3, figsize=(12,6))

ax = axes[0]
im = ax.imshow(l_x.T, interpolation='none', extent=axStyle[0],
    origin="lower", cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1)
cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
    ticks=[vmin, 0, vmax])
cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
ax.set_title(r'$\ell(x)$', fontsize=18)

ax = axes[1]
im = ax.imshow(g_x.T, interpolation='none', extent=axStyle[0],
    origin="lower", cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1)
cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
    ticks=[vmin, 0, vmax])
cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
ax.set_title(r'$g(x)$', fontsize=18)

ax = axes[2]
im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
    origin="lower", cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1)
cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
    ticks=[vmin, 0, vmax])
cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
ax.set_title(r'$v(x)$', fontsize=18)

for ax in axes:
    env.plot_target_failure_set(ax=ax)
    env.plot_formatting(ax=ax)
plt.tight_layout()
plt.show()
print()


# In[9]:


# Run one trial
for i in range(1):
    print('\n== {} =='.format(i))
    env.reset(random_init=False)
    for t in range(1):
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)
        print(obs.shape)
#         print(obs)
        state = info['state']

        # Debug
        x, y, yaw = state
        l_x = info['l_x']
        g_x = info['g_x']

        print('[{}] after a({:d}), x: {:.3f}, y: {:.3f}, yaw: {:.3f}, l_x: {:.3f}, g_x: {:.3f}, d: {}'.format(
            t, action, x, y, yaw, l_x, g_x, done))
        if True:
#             plt.imshow( (255*obs).astype(np.uint8) )
            plt.imshow( obs[0], cmap='gray')
            plt.pause(.25)
            plt.close()
        if done:
            break


# ## AGENT

# In[5]:


print("== Agent Information ==")
maxUpdates = 400000
updateTimes = 20
updatePeriod = int(maxUpdates / updateTimes)
maxSteps = 250
device = 'cpu'

EPS_PERIOD = updatePeriod
EPS_RESET_PERIOD = maxUpdates

args = types.SimpleNamespace()
args.lrC = 1e-3
args.gamma = 0.9999
GAMMA_END = 0.9999
args.memoryCapacity = 10000
args.architecture = [100, 20]
args.actType = 'ReLU'
args.reward = -1
args.penalty = 1
args.terminalType = 'g'

CONFIG = dqnConfig(
    DEVICE=device,
    ENV_NAME=env_name,
    SEED=0,
    MAX_UPDATES=maxUpdates,
    MAX_EP_STEPS=maxSteps,
    BATCH_SIZE=64,
    MEMORY_CAPACITY=args.memoryCapacity,
    ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType,
    GAMMA=args.gamma,
    GAMMA_PERIOD=updatePeriod,
    GAMMA_END=GAMMA_END,
    EPS_PERIOD=EPS_PERIOD,
    EPS_DECAY=0.6,
    EPS_RESET_PERIOD=EPS_RESET_PERIOD,
    LR_C=args.lrC ,
    LR_C_PERIOD=updatePeriod,
    LR_C_DECAY=0.8,
    MAX_MODEL=50)

# for key, value in CONFIG.__dict__.items():
#     if key[:1] != '_': print(key, value)


# In[6]:


def getModelInfo(path):
    dataFolder = os.path.join('scratch', path)
    modelFolder = os.path.join(dataFolder, 'model')
    configPath = os.path.join(modelFolder, 'CONFIG.pkl')
    with open(configPath, 'rb') as fp:
        config = pickle.load(fp)
    config.DEVICE = 'cpu'
    
    trainPath = os.path.join(dataFolder, 'train.pkl')
    with open(trainPath, 'rb') as fp:
        trainDict = pickle.load(fp)

    dimList = [stateDim] + config.ARCHITECTURE + [actionNum]
    return dataFolder, config, dimList, trainDict


# In[9]:


#== AGENT ==
learnFromScratch = True

if learnFromScratch:
    outFolder = os.path.join('scratch', 'car-DDQN-image', doneType)
    print(outFolder)
    os.makedirs(outFolder, exist_ok=True)

    dimList = CONFIG.ARCHITECTURE + [actionNum]
    kernel_sz = [5, 5]
    n_channel = [1, 6, 16]
    agent = DDQN_image(CONFIG, actionSet, dimList, img_sz, kernel_sz, n_channel,
                terminalType=args.terminalType, use_sm=False)
    print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
    print("Critic is using cuda: ", next(agent.Q_network.parameters()).is_cuda)


# In[10]:


lossArray = agent.initQ(env, 20, outFolder, num_warmup_samples=4096, vmin=-1, vmax=1, plotFigure=False, storeFigure=False)


# In[11]:


lossList = lossArray.reshape(-1)
print(lossList.shape)
fig, ax = plt.subplots(1,1, figsize=(4, 4))
tmp = np.arange(500, lossList.shape[0])
ax.plot(tmp, lossList[tmp], 'b-')
ax.set_xlabel('Iteration', fontsize=18)
ax.set_ylabel('Loss', fontsize=18)
plt.tight_layout()
plt.show()


# In[12]:


state = np.array([1.8, 0., 0.])
g_x = env.safety_margin(state)
l_x = env.target_margin(state)
obs = env._get_obs(state)
print(obs)
fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
ax.imshow(obs[0], cmap='Greys')
plt.close()
obsTensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)

v = agent.Q_network(obsTensor)
v = torch.mean(v, dim=1, keepdim=True)
tmp = max(l_x, g_x)
valueTensor = torch.FloatTensor([tmp]).reshape(1, -1).to(agent.device)

from torch.nn.functional import mse_loss, smooth_l1_loss
print(v, valueTensor)
loss = mse_loss(input=v, target=valueTensor, reduction='mean')
print(loss)