# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
#           Allen Z. Ren ( allen.ren@princeton.edu )

# Examples:
    # RA: python3 sim_car_one_SAC_image.py -sf -of scratch -n 999 -mc 50000
    # python3 sim_car_one_SAC_image.py -sf -of scratch -mu 200 -cp 100 -ut 10 -nx 11 -sm -mc 1000 -m safety -n tmp
import pretty_errors
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import argparse
import gym

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")

#= AGENT
from RARL.utils import save_obj
from RARL.SAC_image_maxEnt_vec import SAC_image_maxEnt_vec
from RARL.config import SACImageMaxEntConfig

#= ENV
from safe_rl.navigation_obs_pb_cont import NavigationObsPBEnvCont
from safe_rl.envs import make_vec_envs

def main(args):

    #== CONFIGURATION ==
    env_name = 'navigation_pac_ra_cont-v0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maxUpdates = args.maxUpdates
    updateTimes = args.updateTimes
    updatePeriod = int(maxUpdates / updateTimes)

    fn = args.name + '-' + args.mode + '-' + args.doneType
    if args.showTime:
        fn = fn + '-' + timestr

    outFolder = os.path.join(args.outFolder, 'car-SAC-image', fn)
    print(outFolder)
    figureFolder = os.path.join(outFolder, 'figure')
    os.makedirs(figureFolder, exist_ok=True)

    # TODO: sample tasks

    #== Environment ==
    print("\n== Environment Information ==")
    render = False
    img_sz = 48
    venv = make_vec_envs('NavigationObsPBEnvCont-v0', args.randomSeed, 
        num_processes=args.num_cpus, device=device, 
        maxSteps=args.maxSteps, maxEvalSteps=args.maxEvalSteps, #!
        render=render, img_H=img_sz, img_W=img_sz,
        fixed_init=args.fixed_init,
        sparse_reward=args.sparse_reward,
        num_traj_per_visual_initial_states=args.num_traj_per_visual_initial_states,
        doneType=args.doneType)
    env = NavigationObsPBEnvCont(maxSteps=args.maxSteps, maxEvalSteps=args.maxEvalSteps, #!
        render=render, img_H=img_sz, img_W=img_sz,
        fixed_init=args.fixed_init,
        sparse_reward=args.sparse_reward,
        num_traj_per_visual_initial_states=args.num_traj_per_visual_initial_states,
        doneType=args.doneType) # for visualization
    stateDim = venv.get_attr('state_dim')[0] 
    actionDim = venv.get_attr('action_dim')[0]
    actionLim = venv.get_attr('action_lim')[0]
    print("State Dimension: {:d}, Action Dimension: {:d}".format(
        stateDim, actionDim))
    venv.env_method('report', indices=[0])   # call the method for one env
    venv.reset()
    env.reset()

    #== AGENT ==
    print("\n== Agent Information ==")
    if args.annealing:
        GAMMA_END = 0.9999
        GAMMA_PERIOD = updatePeriod
        LR_Al_PERIOD = int(updatePeriod/10)
    else:
        GAMMA_END = args.gamma
        GAMMA_PERIOD = maxUpdates
        LR_Al_PERIOD = updatePeriod

    MLP_DIM = {'critic':args.dim_critic, 'actor':args.dim_actor}
    ACTIVATION = {'critic':args.actType, 'actor':args.actType}

    CONFIG = SACImageMaxEntConfig(
        # Latent
        LATENT_DIM=args.latent_dim,
        LATENT_PRIOR_STD=args.latent_prior_std,
        AUG_REWARD_RANGE=args.aug_reward_range,
        FIT_FREQ=args.fit_freq,
        # Environment
        FIXED_INIT=args.fixed_init,
        ENV_NAME=env_name,
        SEED=args.randomSeed,
        IMG_SZ=img_sz,
        ACTION_MAG=float(actionLim),
        ACTION_DIM=actionDim,
        # Agent
        DEVICE=device,
        # Training Setting
        MAX_UPDATES=maxUpdates,
        MAX_EP_STEPS=args.maxSteps,
        MEMORY_CAPACITY=args.memoryCapacity,
        BATCH_SIZE=args.batchSize,
        ALPHA=args.alpha,
        LEARN_ALPHA=args.learn_alpha,
        # RL Type
        MODE=args.mode,
        TERMINAL_TYPE=args.terminalType,
        # NN Architecture
        USE_BN=args.batch_norm,
        USE_SM=args.softmax,
        KERNEL_SIZE=args.kernel_sz,
        N_CHANNEL=args.n_channel,
        MLP_DIM=MLP_DIM,
        ACTIVATION=ACTIVATION,
        # Learning Rate and Discount Factor Scheduler
        GAMMA=args.gamma,
        GAMMA_PERIOD=GAMMA_PERIOD,
        GAMMA_END=GAMMA_END,
        LR_D=args.learningRate, # discriminator
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

    #== Learning ==
    agent = SAC_image_maxEnt_vec(CONFIG)
    trainRecords, trainProgress = agent.learn(
        venv, env, warmupBuffer=True, warmupBufferRatio=args.warmupBufferRatio,
        warmupQ=False, MAX_UPDATES=maxUpdates, 
        # MAX_EP_STEPS=args.maxSteps, MAX_EVAL_EP_STEPS=args.maxEvalSteps,
        minStepBeforeOptimize=args.minStepBeforeOptimize,
        optimizeFreq=args.optimize_freq,
        numUpdatePerOptimize=args.num_update_per_optimize,
        vmin=-0.5, vmax=0.5, numRndTraj=100,
        checkPeriod=args.checkPeriod, outFolder=outFolder,
        plotFigure=args.plotFigure, storeFigure=args.storeFigure)
        # useVis=args.visdom, visEnvName=args.outFolder.split('/')[-1])

    trainDict = {}
    trainDict['trainRecords'] = trainRecords
    trainDict['trainProgress'] = trainProgress
    filePath = os.path.join(outFolder, 'train')
    save_obj(trainDict, filePath)

if __name__ == "__main__":

    #== ARGS ==
    parser = argparse.ArgumentParser()

    # environment parameters
    parser.add_argument("-nc", "--num_cpus",     help="number of threads",
        default=8,      type=int)
    parser.add_argument("-dt",  "--doneType",       help="when to raise done flag",
        default='TF',  type=str)
    parser.add_argument("-rnd", "--randomSeed",     help="random seed",
        default=0,      type=int)
    parser.add_argument("-ms",  "--maxSteps",       help="maximum steps",
        default=50,    type=int)
    parser.add_argument("-mes", "--maxEvalSteps",   help="maximum eval steps",
        default=100,    type=int)
    parser.add_argument("-ts",  "--targetScaling",  help="scaling of ell",
        default=1.,     type=float)
    parser.add_argument("-fi",   "--fixed_init",    help="fixed initialization",
        action="store_true")
    parser.add_argument("-sr",   "--sparse_reward",    help="sparse reward",
        action="store_true")

    # training scheme
    parser.add_argument("-w",   "--warmup",             help="warmup Q-network",
        action="store_true")
    parser.add_argument("-wi",  "--warmupIter",         help="warmup iteration",
        default=10000,  type=int)
    parser.add_argument("-wbr", "--warmupBufferRatio",  help="warmup buffer ratio",
        default=1.0, type=float)
    parser.add_argument("-mu",  "--maxUpdates",         help="maximal #gradient updates",
        default=800000, type=int)
    parser.add_argument("-ut",  "--updateTimes",        help="hyper-param. update times",
        default=20,     type=int)
    parser.add_argument("-mc",  "--memoryCapacity",     help="memoryCapacity",
        default=50000,  type=int)
    parser.add_argument("-cp",  "--checkPeriod",        help="check period",
        default=1000,  type=int)
    parser.add_argument("-bs",  "--batchSize",          help="batch size",
        default=128,  type=int)

    # hyper-parameters
    parser.add_argument("-a",   "--annealing",                  help="gamma annealing",
        action="store_true")
    parser.add_argument("-lr",  "--learningRate",               help="learning rate",
        default=1e-3,   type=float)
    parser.add_argument("-lrd", "--learningRateDecay",          help="learning rate decay",
        default=1.0,   type=float)
    parser.add_argument("-g",   "--gamma",                      help="contraction coeff.",
        default=0.999, type=float)
    parser.add_argument("-al",  "--alpha",                      help="alpha",
        default=0.2, type=float)
    parser.add_argument("-lal", "--learn_alpha",                help="learn alpha",
        action="store_true")
    parser.add_argument("-ues", "--optimize_freq",              help="optimization freq.",
        default=100, type=int)
    parser.add_argument("-nmo", "--num_update_per_optimize",    help="#updates per opt.",
        default=100, type=int)
    parser.add_argument("-arr", "--aug_reward_range",    help="aug reward range",
        default=0.01, type=float)
    parser.add_argument("-ff", "--fit_freq",    help="number of updates for discriminator", default=20, type=int)
    parser.add_argument("-msbo", "--minStepBeforeOptimize",    help="number of steps for initial", default=10000, type=int)

    # NN architecture
    parser.add_argument("-sm",  "--softmax",        help="spatial softmax",
        action="store_true")
    parser.add_argument("-bn",  "--batch_norm",     help="batch normalization",
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
    parser.add_argument("-ld", "--latent_dim",    help="latent dimension",
        default=2, type=int)
    parser.add_argument("-lps", "--latent_prior_std",    help="latent prior std",
        default=1.0, type=float)

    # RL type
    parser.add_argument("-m",   "--mode",           help="mode",
        default='RA',   type=str)
    parser.add_argument("-tt",  "--terminalType",   help="terminal value",
        default='g',    type=str)

    # file
    parser.add_argument("-vis",  "--visdom",        help="use Visdom",
        action="store_true")
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
    parser.add_argument("-nt", "--num_traj_per_visual_initial_states", help="",
        default=10, type=int)

    args = parser.parse_args()
    print(args)

    main(args)
