# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#           Allen Z. Ren ( allen.ren@princeton.edu )

# Examples:
    # RA: python3 sim_shielding_joint.py -sf -of scratch -n 999 -mc 50000
    # python3 sim_shielding_joint.py -sf -of scratch -lal -ln -ms 200 -cp 10 -ut 10 -mc 1000 -ep 1 -rp 1 -n tmp

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import argparse
import copy
from types import SimpleNamespace

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")

#= AGENT
from RARL.utils import save_obj
from RARL.policyShieldingJoint import PolicyShieldingJoint
from RARL.config import NNConfig, TrainingConfig

#= ENV
from safe_rl.navigation_obs_pb_cont import NavigationObsPBEnvCont

# region: == ARGS ==
parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument("-rnd", "--randomSeed",     help="random seed",
    default=0,      type=int)
parser.add_argument("-mts", "--maxTrainSteps",  help="max steps in training episode",
    default=500,    type=int)
parser.add_argument("-mes", "--maxEvalSteps",   help="max steps in evaluation episode",
    default=250,    type=int)
parser.add_argument("-fi",  "--fixed_init",     help="layer normalization",
    action="store_true")

# training scheme
parser.add_argument("-wbr", "--warmupBufferRatio",  help="warmup buffer ratio",
    default=1.0,    type=float)
parser.add_argument("-ms",  "--maxSteps",         help="maximal #gradient updates",
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
    default='value', type=str,   choices=['simulator', 'value'])

# hyper-parameters
parser.add_argument("-lr",  "--learningRate",               help="learning rate",
    default=1e-3,   type=float)
parser.add_argument("-lrd", "--learningRateDecay",          help="learning rate decay",
    default=0.9,    type=float)
parser.add_argument("-al",  "--alpha",                      help="alpha",
    default=0.2,    type=float)
parser.add_argument("-ep",  "--epsPeriod",                  help="period of eps update",
    default=100,    type=int)
parser.add_argument("-ed",  "--epsDecay",                   help="eps decay",
    default=0.8,    type=float)
parser.add_argument("-rp",  "--rhoPeriod",                  help="period of rho update",
    default=100,    type=int)
parser.add_argument("-rd",  "--rhoDecay",                   help="rho decay",
    default=0.9,    type=float)
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
maxSteps = args.maxSteps
updateTimes = args.updateTimes
updatePeriod = int(maxSteps / updateTimes)

fn = args.name + '-' + args.shieldType
if args.fixed_init:
    fn = fn + '-fix'
if args.showTime:
    fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'car-joint', fn)
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
GAMMA_PERIOD = maxSteps
LR_Al_PERIOD = updatePeriod

MLP_DIM    = {'critic': args.dim_critic, 'actor': args.dim_actor}
ACTIVATION = {'critic': args.actType,    'actor': args.actType}

CONFIG_TRAIN_PERF = TrainingConfig(
    # Environment
    ENV_NAME=env_name,
    SEED=args.randomSeed,
    IMG_SZ=img_sz,
    ACTION_MAG=float(actionLim),
    ACTION_DIM=actionDim,
    OBS_CHANNEL=env.num_img_channel,
    # Agent
    MODE='performance',
    DEVICE=device,
    # Training Setting
    TRAIN_BACKUP=False,
    MAX_UPDATES=maxSteps,
    MAX_EP_STEPS=args.maxTrainSteps,
    MEMORY_CAPACITY=args.memoryCapacity,
    BATCH_SIZE=args.batchSize,
    ALPHA=args.alpha,
    LEARN_ALPHA=args.learn_alpha,
    TAU=0.01,
    MAX_MODEL=50,
    # Learning Rate and Discount Factor Scheduler
    GAMMA=0.99,
    GAMMA_PERIOD=GAMMA_PERIOD,
    GAMMA_END=0.99,
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

CONFIG_TRAIN_BACKUP = copy.deepcopy(CONFIG_TRAIN_PERF)
CONFIG_TRAIN_BACKUP.MODE = 'safety'
CONFIG_TRAIN_BACKUP.GAMMA = 0.999
CONFIG_TRAIN_BACKUP.GAMMA_END = 0.999

CONFIG_ARCH = NNConfig(
    USE_BN=False,
    USE_LN=args.layer_norm,
    USE_SM=True,
    KERNEL_SIZE=args.kernel_sz,
    N_CHANNEL=args.n_channel,
    MLP_DIM=MLP_DIM,
    ACTIVATION=ACTIVATION
)

CONFIG = SimpleNamespace()
CONFIG.DEVICE = device
CONFIG.EPS = 0.
CONFIG.EPS_PERIOD = args.epsPeriod
CONFIG.EPS_DECAY = args.epsDecay
CONFIG.EPS_END = 1.
CONFIG.RHO = 0.5
CONFIG.RHO_PERIOD = args.rhoPeriod
CONFIG.RHO_DECAY = args.rhoDecay
CONFIG.RHO_END = 0.1
CONFIG.MEMORY_CAPACITY = args.memoryCapacity
CONFIG.BATCH_SIZE=args.batchSize

CONFIG_PERFORMANCE = {}
CONFIG_PERFORMANCE['train'] = CONFIG_TRAIN_PERF
CONFIG_PERFORMANCE['arch'] = CONFIG_ARCH

CONFIG_BACKUP = {}
CONFIG_BACKUP['train'] = CONFIG_TRAIN_BACKUP
CONFIG_BACKUP['arch'] = CONFIG_ARCH

agent = PolicyShieldingJoint(CONFIG, CONFIG_PERFORMANCE, CONFIG_BACKUP)
print('\nTotal parameters in actor: {}'.format(
    sum(p.numel() for p in agent.performance.actor.parameters() if p.requires_grad) ))
print("We want to use: {}, and Agent uses: {}".format(device, agent.performance.device))
print("Critic is using cuda: ", next(agent.performance.critic.parameters()).is_cuda)

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
    MAX_STEPS=maxSteps, MAX_EP_STEPS=args.maxTrainSteps,
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

    data = trainProgress[0][:, 0]
    ax = axes[1]
    x = np.arange(data.shape[0]) + 1
    ax.plot(x, data, 'b-o')
    ax.set_xlabel('Index', fontsize=18)
    ax.set_xticks(x)
    ax.set_title('Success (Performance)', fontsize=18)
    ax.set_xlim(left=1, right=data.shape[0])
    ax.set_ylim(0, 0.8)

    data = trainProgress[1][:, 0]
    ax = axes[1]
    x = np.arange(data.shape[0]) + 1
    ax.plot(x, data, 'b-o')
    ax.set_xlabel('Index', fontsize=18)
    ax.set_xticks(x)
    ax.set_title('Success (Backup)', fontsize=18)
    ax.set_xlim(left=1, right=data.shape[0])
    ax.set_ylim(0, 0.8)

    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'train_success.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()
    # endregion

save_obj(trainDict, filePath)
# endregion