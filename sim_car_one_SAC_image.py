# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#           Allen Z. Ren ( allen.ren@princeton.edu )

# This experiment runs double deep Q-network with the discounted reach-avoid
# Bellman equation (DRABE) proposed in [RSS21] on a 3-dimensional Dubins car
# problem. We use this script to generate Fig. 5 in the paper.

# Examples:
    # RA: python3 sim_car_one_SAC_image.py -sf -of scratch -n 999 -mc 50000
    # python3 sim_car_one_SAC_image.py -sf -of scratch -mu 300 -cp 140 -ut 10 -nx 21 -sm -mc 1000 -n tmp

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import pretty_errors
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import argparse

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")

#= AGENT
from RARL.utils import save_obj
from RARL.SAC_image import SAC_image
from RARL.config import SACImageConfig

#= ENV
from safe_rl.navigation_obs_pb_cont import NavigationObsPBEnvCont

#== ARGS ==
parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument("-dt",  "--doneType",       help="when to raise done flag",
    default='fail',  type=str)
parser.add_argument("-rnd", "--randomSeed",     help="random seed",
    default=0,      type=int)
parser.add_argument("-ms",  "--maxSteps",       help="maximum steps",
    default=20,    type=int)
parser.add_argument("-mes",  "--maxEvalSteps",   help="maximum eval steps",
    default=100,    type=int)
parser.add_argument("-ts",  "--targetScaling",  help="scaling of ell",
    default=1.,     type=float)

# training scheme
parser.add_argument("-w",   "--warmup",         help="warmup Q-network",
    action="store_true")
parser.add_argument("-wi",  "--warmupIter",     help="warmup iteration",
    default=10000,  type=int)
parser.add_argument("-wbr",  "--warmupBufferRatio",  help="warmup buffer ratio",
    default=1.0, type=float)
parser.add_argument("-mu",  "--maxUpdates",     help="maximal #gradient updates",
    default=800000, type=int)
parser.add_argument("-ut",  "--updateTimes",    help="hyper-param. update times",
    default=20,     type=int)
parser.add_argument("-mc",  "--memoryCapacity", help="memoryCapacity",
    default=50000,  type=int)
parser.add_argument("-cp",  "--checkPeriod",    help="check period",
    default=1000,  type=int)
parser.add_argument("-bs",  "--batchSize",      help="batch size",
    default=128,  type=int)

# hyper-parameters
parser.add_argument("-a",   "--annealing",      help="gamma annealing",
    action="store_true")
parser.add_argument("-sm",  "--softmax",        help="spatial softmax",
    action="store_true")
parser.add_argument("-bn",  "--batch_norm",     help="batch normalization",
    action="store_true")
parser.add_argument("-arc", "--mlp_dim",   help="NN architecture",
    default=[[64, 64], [128, 128]],     nargs="*", type=int)
parser.add_argument("-nch", "--n_channel",      help="NN architecture",
    default=[16, 32, 64],   nargs="*", type=int)
parser.add_argument("-ksz", "--kernel_sz",      help="NN architecture",
    default=[5, 5, 3],      nargs="*", type=int)
parser.add_argument("-lr",  "--learningRate",   help="learning rate",
    default=1e-3,   type=float)
parser.add_argument("-lrd",  "--learningRateDecay",   help="learning rate decay",
    default=1.0,   type=float)
parser.add_argument("-g",   "--gamma",          help="contraction coeff.",
    default=0.999, type=float)
parser.add_argument("-act", "--actType",        help="activation type",
    default='ReLU', type=str)
parser.add_argument("-al",   "--alpha",          help="alpha",
    default=0.2, type=float)
parser.add_argument("-lal",  "--learn_alpha",    help="learn alpha",
    action="store_true")
parser.add_argument("-ues",  "--optimize_freq",    help="optimize after  some samples", default=100, type=int)
parser.add_argument("-nmo",  "--num_update_per_optimize",    help="number of updates per optimize", default=128, type=int)

# RL type
parser.add_argument("-ur",   "--use_RA",        help="mode",
    action="store_true")
parser.add_argument("-tt",  "--terminalType",   help="terminal value",
    default='g',    type=str)

# file
parser.add_argument("-nx",  "--nx",             help="check period",
    default=101, type=int)
parser.add_argument("-st",  "--showTime",       help="show timestr",
    action="store_true")
parser.add_argument("-n",   "--name",           help="extra name",
    default='', type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",
    default='/scratch/gpfs/kaichieh/',  type=str)
parser.add_argument("-pf",  "--plotFigure",     help="plot figures",
    action="store_true")
parser.add_argument("-sf",  "--storeFigure",    help="store figures",
    action="store_true")

args = parser.parse_args()
print(args)


#== CONFIGURATION ==
env_name = 'navigation_pac_ra_disc-v0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)

fn = args.name + '-' + args.doneType
if args.showTime:
    fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'car-DDQN-image', fn)
print(outFolder)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)

#== Environment ==
render = False
img_sz = 48
env = NavigationObsPBEnvCont(render=render, img_H=img_sz, img_W=img_sz,
        doneType=args.doneType)
env.seed(args.randomSeed)

stateDim = env.state_dim
actionDim = env.action_dim
actionLim = env.action_lim
print("State Dimension: {:d}, Action Dimension: {:d}".format(
    stateDim, actionDim))
env.report()

#== Get and Plot max{l_x, g_x} ==
if args.plotFigure or args.storeFigure:
    nx = 101
    ny = nx
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

    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'env.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()

#== AGENT ==
print("== Agent Information ==")
if args.annealing:
    GAMMA_END = 0.9999
else:
    GAMMA_END = args.gamma


CONFIG = SACImageConfig(
    ENV_NAME=env_name,
    DEVICE=device,
    SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates,
    MAX_EP_STEPS=args.maxSteps,
    BATCH_SIZE=args.batchSize,
    MEMORY_CAPACITY=args.memoryCapacity,
    GAMMA=args.gamma,
    GAMMA_PERIOD=updatePeriod,
    GAMMA_END=GAMMA_END,
    LR_C=args.learningRate,
    LR_C_PERIOD=updatePeriod,
    LR_C_DECAY=args.learningRateDecay,
    MAX_MODEL=50,
    LEARN_ALPHA=args.learn_alpha,
    ALPHA=args.alpha,
    IMG_SZ=img_sz,
    MLP_DIM=args.mlp_dim,
    ACTIVATION=args.actType,
    KERNEL_SIZE=args.kernel_sz,
    N_CHANNEL=args.n_channel,
    USE_BN=args.batch_norm,
    USE_SM=args.softmax,
    ACTION_MAG=float(actionLim),
    ACTION_DIM=actionDim,
    USE_RA=args.use_RA,
)

# for key, value in CONFIG.__dict__.items():
#     if key[:1] != '_': print(key, value)
agent = SAC_image(CONFIG, terminalType=args.terminalType)
print(f'Total parameters in actor: {sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)}')
print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
print("Critic is using cuda: ", next(agent.critic.parameters()).is_cuda)

if args.warmup:
    print("\n== Warmup Q ==")
    lossArray = agent.initQ(env, 20, outFolder, num_warmup_samples=4096,
        vmin=-1, vmax=1, plotFigure=args.plotFigure, storeFigure=args.storeFigure)

    if args.plotFigure or args.storeFigure:
        lossList = lossArray.reshape(-1)
        fig, ax = plt.subplots(1,1, figsize=(4, 4))
        tmp = np.arange(500, args.warmupIter)
        # tmp = np.arange(args.warmupIter)
        ax.plot(tmp, lossList[tmp], 'b-')
        ax.set_xlabel('Iteration', fontsize=18)
        ax.set_ylabel('Loss', fontsize=18)
        plt.tight_layout()

        if args.storeFigure:
            figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
            fig.savefig(figurePath)
        if args.plotFigure:
            plt.show()
            plt.pause(0.001)
        plt.close()

# agent.initBuffer(env)
# transitions = agent.memory.sample(5)
# batch = Transition(*zip(*transitions))
# non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x = \
#             agent.unpack_batch(batch)
# print(action)
# print(l_x)


#== Learning ==
trainRecords, trainProgress = agent.learn(
    env, warmupBuffer=True, warmupBufferRatio=args.warmupBufferRatio,
    warmupQ=False, MAX_UPDATES=maxUpdates, MAX_EP_STEPS=args.maxSteps,
    MAX_EVAL_EP_STEPS=args.maxEvalSteps,
    optimizeFreq=args.optimize_freq, 
    numUpdatePerOptimize=args.num_update_per_optimize,
    vmin=-0.5, vmax=0.5, numRndTraj=100,
    checkPeriod=args.checkPeriod, outFolder=outFolder,
    plotFigure=args.plotFigure, storeFigure=args.storeFigure)

trainDict = {}
trainDict['trainRecords'] = trainRecords
trainDict['trainProgress'] = trainProgress
filePath = os.path.join(outFolder, 'train')

if args.plotFigure or args.storeFigure:
    # region: loss
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    data = trainRecords
    ax = axes[0]
    ax.plot(data, 'b:')
    ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
    ax.set_xticks(np.linspace(0, maxUpdates, 5))
    ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
    ax.set_title('loss_critic', fontsize=18)
    ax.set_xlim(left=0, right=maxUpdates)

    data = trainProgress[:, 0]
    ax = axes[1]
    x = np.arange(data.shape[0]) + 1
    ax.plot(x, data, 'b-o')
    ax.set_xlabel('Index', fontsize=18)
    ax.set_xticks(x)
    ax.set_title('Success Rate', fontsize=18)
    ax.set_xlim(left=1, right=data.shape[0])
    ax.set_ylim(0, 0.8)

    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'train_loss_success.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()
    # endregion

    # region: value_rollout_action
    # idx = np.argmax(trainProgress[:, 0]) + 1
    # successRate = np.amax(trainProgress[:, 0])
    # print('We pick model with success rate-{:.3f}'.format(successRate))
    idx = trainProgress.shape[0]
    agent.restore(idx*args.checkPeriod, outFolder)
    policy = lambda obs: agent.select_action(obs, explore=False)[1]

    nx = args.nx
    ny = nx
    xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
    ys = np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

    resultMtx  = np.empty((nx, ny), dtype=int)
    actDistMtx = np.empty((nx, ny), dtype=int)
    it = np.nditer(resultMtx, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]

        state = np.array([x, y, 0.])
        obs = env._get_obs(state)
        obsTensor = torch.FloatTensor(obs).to(agent.device).unsqueeze(0)
        action_index = agent.Q_network(obsTensor).min(dim=1)[1].cpu().detach().numpy()
        actDistMtx[idx] = action_index

        _, result, _, _ = env.simulate_one_trajectory(
                            policy, T=250, state=state, toEnd=False)
        resultMtx[idx] = result
        it.iternext()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axStyle = env.get_axes()

    #= Action
    ax = axes[2]
    im = ax.imshow(actDistMtx.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=0, vmax=1, zorder=-1)
    ax.set_xlabel('Action', fontsize=24)

    #= Rollout
    ax = axes[1]
    im = ax.imshow(resultMtx.T != 1, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=0, vmax=1, zorder=-1)
    env.plot_trajectories(policy, ax, num_rnd_traj=5, theta=0.,
        toEnd=False, c='w', lw=1.5, T=250)
    ax.set_xlabel('Rollout RA', fontsize=24)

    #= Value
    ax = axes[0]
    v, xs, ys = env.get_value(agent.Q_network, agent.device, theta=0, nx=nx, ny=ny)
    im = ax.imshow(v.T, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=-0.5, vmax=0.5, zorder=-1)
    CS = ax.contour(xs, ys, v.T, levels=[0], colors='k', linewidths=2,
        linestyles='dashed')
    ax.set_xlabel('Value', fontsize=24)

    for ax in axes:
        env.plot_target_failure_set(ax)
        env.plot_formatting(ax)

    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'value_rollout_action.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()
    # endregion

    trainDict['resultMtx'] = resultMtx
    trainDict['actDistMtx'] = actDistMtx

save_obj(trainDict, filePath)