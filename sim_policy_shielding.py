# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#           Allen Z. Ren ( allen.ren@princeton.edu )

# Examples:
    # RA: python3 sim_policy_shielding.py -sf -of scratch -n 999 -mc 50000
    # python3 sim_policy_shielding.py -sf -lal -ln -of scratch -mu 200 -cp 100 -ut 10 -mc 1000 -n tmp

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

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
from RARL.policyShielding import PolicyShielding
from RARL.config import NNConfig, TrainingConfig

#= ENV
from safe_rl.navigation_obs_pb_cont import NavigationObsPBEnvCont

# region: == ARGS ==
parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument("-rnd", "--randomSeed",     help="random seed",
    default=0,      type=int)
parser.add_argument("-ms",  "--maxSteps",       help="maximum steps",
    default=500,    type=int)
parser.add_argument("-mes", "--maxEvalSteps",   help="maximum eval steps",
    default=250,    type=int)
parser.add_argument("-fi",  "--fixed_init",     help="layer normalization",
    action="store_true")

# training scheme
parser.add_argument("-wbr", "--warmupBufferRatio",  help="warmup buffer ratio",
    default=1.0,    type=float)
parser.add_argument("-mu",  "--maxUpdates",         help="maximal #gradient updates",
    default=200000, type=int)
parser.add_argument("-ut",  "--updateTimes",        help="hyper-param. update times",
    default=20,     type=int)
parser.add_argument("-mc",  "--memoryCapacity",     help="memoryCapacity",
    default=20000,  type=int)
parser.add_argument("-cp",  "--checkPeriod",        help="check period",
    default=10000,  type=int)
parser.add_argument("-bs",  "--batchSize",          help="batch size",
    default=128,    type=int)
parser.add_argument("-sht", "--shieldType",         help="when to raise shield flag",
    default='none', type=str,   choices=['none', 'simulator', 'value'])

# hyper-parameters
parser.add_argument("-lr",  "--learningRate",               help="learning rate",
    default=1e-3,   type=float)
parser.add_argument("-lrd", "--learningRateDecay",          help="learning rate decay",
    default=0.9,    type=float)
parser.add_argument("-g",   "--gamma",                      help="contraction coeff.",
    default=0.99,   type=float)
parser.add_argument("-al",  "--alpha",                      help="alpha",
    default=0.2,    type=float)
parser.add_argument("-ues", "--optimize_freq",              help="optimization freq.",
    default=100,    type=int)
parser.add_argument("-nmo", "--num_update_per_optimize",    help="#updates per opt.",
    default=100,    type=int)
parser.add_argument("-lal", "--learn_alpha",                help="learn alpha",
    action="store_true")

# NN architecture:
parser.add_argument("-ln",  "--layer_norm",     help="layer normalization",
    action="store_true")
parser.add_argument("-d_c", "--dim_critic",     help="critic mlp dimension",
    default=[128, 128], nargs="*", type=int)
parser.add_argument("-d_a", "--dim_actor",      help="actor mlp dimension",
    default=[64, 64],   nargs="*", type=int)
parser.add_argument("-nch", "--n_channel",      help="NN architecture",
    default=[16, 32, 64],   nargs="*", type=int)
parser.add_argument("-ksz", "--kernel_sz",      help="NN architecture",
    default=[5, 5, 3],      nargs="*", type=int)
parser.add_argument("-act", "--actType",        help="activation type",
    default='ReLU', type=str)

# file
# parser.add_argument("-nx",  "--nx",             help="check period",
#     default=101, type=int)
parser.add_argument("-st",  "--showTime",       help="show timestr",
    action="store_true")
parser.add_argument("-n",   "--name",           help="extra name",
    default='', type=str)
parser.add_argument("-of",  "--outFolder",      help="output file",
    default='/scratch/gpfs/kaichieh/',  type=str)
# parser.add_argument("-bf",  "--backupFolder",   help="backup critic/actor",
#     default='/scratch/gpfs/kaichieh/',  type=str)
parser.add_argument("-pf",  "--plotFigure",     help="plot figures",
    action="store_true")
parser.add_argument("-sf",  "--storeFigure",    help="store figures",
    action="store_true")

args = parser.parse_args()
print(args)
# endregion


# region: == CONFIGURATION ==
env_name = 'navigation_pac_ra_cont-v0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)

fn = args.name + '-' + args.shieldType
if args.fixed_init:
    fn = fn + '-fix'
if args.showTime:
    fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'car-shield', fn)
print(outFolder)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)
# endregion


# region: == Environment ==
print("\n== Environment Information ==")
img_sz = 48
env = NavigationObsPBEnvCont(
    img_H=img_sz,
    img_W=img_sz,
    num_traj_per_visual_initial_states=1,
    fixed_init=args.fixed_init,
    sparse_reward=False,
    useRGB=True,
    render=False,
    doneType='fail'
)
env.seed(args.randomSeed)

stateDim = env.state_dim
actionDim = env.action_dim
actionLim = env.action_lim
env.report()
env.reset()

#= Plot Env
if args.plotFigure or args.storeFigure:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
    env.plot_target_failure_set(ax)
    env.plot_formatting(ax, labels=None, fsz=16)

    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'env.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()
# endregion


# region: == AGENT ==
print("\n== Agent Information ==")
GAMMA_END = args.gamma
GAMMA_PERIOD = maxUpdates
LR_Al_PERIOD = updatePeriod

MLP_DIM    = {'critic': args.dim_critic, 'actor': args.dim_actor}
ACTIVATION = {'critic': args.actType,    'actor': args.actType}

CONFIG = TrainingConfig(
    # Environment
    ENV_NAME=env_name,
    SEED=args.randomSeed,
    IMG_SZ=img_sz,
    ACTION_MAG=float(actionLim),
    ACTION_DIM=actionDim,
    OBS_CHANNEL=env.num_img_channel,
    # Agent
    DEVICE=device,
    # Training Setting
    TRAIN_BACKUP=False,
    MAX_UPDATES=maxUpdates,
    MAX_EP_STEPS=args.maxSteps,
    MEMORY_CAPACITY=args.memoryCapacity,
    BATCH_SIZE=args.batchSize,
    ALPHA=args.alpha,
    LEARN_ALPHA=args.learn_alpha,
    TAU=0.01,
    MAX_MODEL=50,
    # Learning Rate and Discount Factor Scheduler
    GAMMA=args.gamma,
    GAMMA_PERIOD=GAMMA_PERIOD,
    GAMMA_END=GAMMA_END,
    LR_C=args.learningRate,
    LR_C_PERIOD=updatePeriod,
    LR_C_END=args.learningRate/10,
    LR_C_DECAY=args.learningRateDecay,
    LR_A=args.learningRate,
    LR_A_PERIOD=updatePeriod,
    LR_A_END=args.learningRate/10,
    LR_A_DECAY=args.learningRateDecay,
    LR_Al=5e-4,
    LR_Al_END=1e-5,
    LR_Al_PERIOD=LR_Al_PERIOD,
    LR_Al_DECAY=0.9,
)

CONFIG_ARCH = NNConfig(
    USE_BN=False,
    USE_LN=args.layer_norm,
    USE_SM=True,
    KERNEL_SIZE=args.kernel_sz,
    N_CHANNEL=args.n_channel,
    MLP_DIM=MLP_DIM,
    ACTIVATION=ACTIVATION
)
# for key, value in CONFIG.__dict__.items():
#     if key[:1] != '_': print(key, value)

agent = PolicyShielding(CONFIG, CONFIG_ARCH, CONFIG_ARCH)
print('Total parameters in actor: {}'.format(
    sum(p.numel() for p in agent.actor.parameters() if p.requires_grad) ))
print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
print("Critic is using cuda: ", next(agent.critic.parameters()).is_cuda)

print("Use pre-trained backup policy:")
backupFolder = os.path.join(
    args.outFolder, 'car-SAC-image', '999-safety-fail', 'model')
agent.restore(200000, backupFolder, 'backup')
# endregion


# region: == Learning ==
print("\n== Learning ==")
shieldDict = {}
shieldDict['Type'] = args.shieldType
if shieldDict['Type'] == 'simulator':
    shieldDict['T_rollout'] = 100
if shieldDict['Type'] == 'value':
    shieldDict['Threshold'] = -0.02

trainRecords, trainProgress, violationRecord = agent.learn(
    env, shieldDict,
    warmupBuffer=True, warmupBufferRatio=args.warmupBufferRatio,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=args.maxSteps,
    # MAX_UPDATES=10000, MAX_EP_STEPS=1000,
    MAX_EVAL_EP_STEPS=args.maxEvalSteps,
    optimizeFreq=args.optimize_freq,
    numUpdatePerOptimize=args.num_update_per_optimize,
    vmin=-0.5, vmax=0.5, numRndTraj=100,
    checkPeriod=args.checkPeriod, outFolder=outFolder,
    plotFigure=args.plotFigure, storeFigure=args.storeFigure)
print('The number of safety violations: {:d}/{:d}'.format(
    violationRecord[-1], len(violationRecord)))
# endregion


# region: == Training Result Dictionary ==
trainDict = {}
trainDict['trainRecords'] = trainRecords
trainDict['trainProgress'] = trainProgress
trainDict['violationRecord'] = violationRecord
filePath = os.path.join(outFolder, 'train')

if args.plotFigure or args.storeFigure:
    # region: loss
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    data = trainRecords[:, 0]
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
    # idx = trainProgress.shape[0]
    # performancePolicyFolder = os.path.join(outFolder, 'model', 'performance')
    # agent.restore(idx*args.checkPeriod, performancePolicyFolder, agentType='performance')
    # policy = lambda obs, z : agent.actor(obs, z)
    # rolloutEndType = 'TF'

    # nx = args.nx
    # ny = nx
    # xs = np.linspace(env.bounds[0,0], env.bounds[0,1], nx)
    # ys = np.linspace(env.bounds[1,0], env.bounds[1,1], ny)

    # resultMtx  = np.empty((nx, ny), dtype=int)
    # actDistMtx = np.empty((nx, ny), dtype=float)
    # valueMtx = np.empty((nx, ny), dtype=float)
    # it = np.nditer(resultMtx, flags=['multi_index'])

    # while not it.finished:
    #     idx = it.multi_index
    #     print(idx, end='\r')
    #     x = xs[idx[0]]
    #     y = ys[idx[1]]

    #     state = np.array([x, y, 0.])
    #     obs = env._get_obs(state)
    #     obsTensor = torch.FloatTensor(obs).to(agent.device).unsqueeze(0)
    #     u = agent.actor(obsTensor).detach()
    #     v = agent.critic(obsTensor, u)[0].cpu().detach().numpy()[0]
    #     actDistMtx[idx] = u.cpu().numpy()[0]
    #     valueMtx[idx] = v

    #     _, result, _, _ = env.simulate_one_trajectory(
    #         policy, T=args.maxEvalSteps, state=state, endType=rolloutEndType)
    #     resultMtx[idx] = result
    #     it.iternext()

    # resultVisMtx = (resultMtx != 1)

    # fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    # axStyle = env.get_axes()

    # #= Action
    # ax = axes[2]
    # im = ax.imshow(actDistMtx.T, interpolation='none', extent=axStyle[0],
    #     origin="lower", cmap='seismic', vmin=-actionLim, vmax=actionLim, zorder=-1)
    # ax.set_xlabel('Action', fontsize=24)

    # #= Rollout
    # ax = axes[1]
    # im = ax.imshow(resultVisMtx.T, interpolation='none', extent=axStyle[0],
    #     origin="lower", cmap='seismic', vmin=0, vmax=1, zorder=-1)
    # env.plot_trajectories(policy, ax, num_rnd_traj=5, theta=0.,
    #     endType=rolloutEndType, c='w', lw=1.5, T=args.maxSteps)
    # ax.set_xlabel('Rollout RA', fontsize=24)

    # #= Value
    # ax = axes[0]
    # # v, xs, ys = env.get_value(q_func, theta=0, nx=nx, ny=ny)
    # im = ax.imshow(valueMtx.T, interpolation='none', extent=axStyle[0],
    #     origin="lower", cmap='seismic', vmin=-0.5, vmax=0.5, zorder=-1)
    # CS = ax.contour(xs, ys, valueMtx.T, levels=[0], colors='k', linewidths=2,
    #     linestyles='dashed')
    # ax.set_xlabel('Value', fontsize=24)

    # for ax in axes:
    #     env.plot_target_failure_set(ax)
    #     env.plot_formatting(ax)

    # fig.tight_layout()
    # if args.storeFigure:
    #     figurePath = os.path.join(figureFolder, 'value_rollout_action.png')
    #     fig.savefig(figurePath)
    # if args.plotFigure:
    #     plt.show()
    #     plt.pause(0.001)
    # plt.close()

    # trainDict['resultMtx'] = resultMtx
    # trainDict['actDistMtx'] = actDistMtx

    # endregion

save_obj(trainDict, filePath)
# endregion