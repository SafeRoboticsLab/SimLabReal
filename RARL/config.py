# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

class config():
    def __init__(self,  ENV_NAME='Pendulum-v0',
                        DEVICE='cpu', SEED=0,
                        MAX_UPDATES=2000000, MAX_EP_STEPS=200,
                        EPSILON=0.95, EPS_END=0.05, EPS_PERIOD=1, EPS_DECAY=0.5,
                        EPS_RESET_PERIOD=100,
                        LR_C=1e-3, LR_C_END=1e-4, LR_C_PERIOD=1, LR_C_DECAY=0.5,
                        GAMMA=0.9, GAMMA_END=0.99999999, GAMMA_PERIOD=200, GAMMA_DECAY=0.5,
                        MEMORY_CAPACITY=10000,
                        BATCH_SIZE=32,
                        RENDER=False,
                        MAX_MODEL=5,
                        ARCHITECTURE=[512, 512, 512],
                        ACTIVATION='Tanh',
                        SKIP=False,
                        REWARD=-1,
                        PENALTY=1):
        """
        __init__

        Args:
            ENV_NAME (str, optional): enironment name.
                Defaults to 'Pendulum-v0'.
            DEVICE (str, optional): on which you want to run your PyTorch model.
                Defaults to 'cpu'.
            SEED (int, optional): random seed.
                Defaults to 0.
            MAX_UPDATES (int, optional): maximal number of gradient updates.
                Defaults to 2000000.
            MAX_EP_STEPS (int, optional): maximal number of steps in an episode.
                Defaults to 200.
            EPSILON (float, optional): control the exploration-exploitation tradeoff.
                Defaults to 0.95.
            EPS_END (float, optional): terminal value of epsilon.
                Defaults to 0.05.
            EPS_PERIOD (int, optional): update period of epsilon.
                Defaults to 1.
            EPS_DECAY (float, optional): multiplicative factor of epsilon.
                Defaults to 0.5.
            LR_C (float, optional): learning rate of critic model.
                Defaults to 1e-3.
            LR_C_END (float, optional): terminal value of LR_C.
                Defaults to 1e-4.
            LR_C_PERIOD (int, optional): update period of LR_C.
                Defaults to 1.
            LR_C_DECAY (float, optional): multiplicative factor of LR_C.
                Defaults to 0.5.
            GAMMA (float, optional): discount factor.
                Defaults to 0.9.
            GAMMA_END (float, optional): terminal value of gamma.
                Defaults to 0.99999999.
            GAMMA_PERIOD (int, optional): update period of gamma.
                Defaults to 200.
            GAMMA_DECAY (float, optional): multiplicative factor of gamma.
                Defaults to 0.5.
            MEMORY_CAPACITY (int, optional): the size of replay buffer.
                Defaults to 10000.
            BATCH_SIZE (int, optional): the number of samples you want to use to update the critic model.
                Defaults to 32.
            RENDER (bool, optional): render the environment or not.
                Defaults to False.
            MAX_MODEL (int, optional): maximal number of models you want to store during the training process.
                Defaults to 5.
            ARCHITECTURE (list, optional): the architecture of the hidden layers of the NN.
                Defaults to [512, 512, 512].
            ACTIVATION (str, optional): the activation function used in the NN.
                Defaults to 'Tanh'.
        """
        self.MAX_UPDATES = MAX_UPDATES
        self.MAX_EP_STEPS = MAX_EP_STEPS

        self.EPSILON = EPSILON
        self.EPS_END = EPS_END
        self.EPS_PERIOD = EPS_PERIOD
        self.EPS_DECAY = EPS_DECAY
        self.EPS_RESET_PERIOD = EPS_RESET_PERIOD

        self.LR_C = LR_C
        self.LR_C_END = LR_C_END
        self.LR_C_PERIOD = LR_C_PERIOD
        self.LR_C_DECAY = LR_C_DECAY

        self.GAMMA = GAMMA
        self.GAMMA_END = GAMMA_END
        self.GAMMA_PERIOD = GAMMA_PERIOD
        self.GAMMA_DECAY = GAMMA_DECAY

        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE

        self.RENDER = RENDER
        self.ENV_NAME = ENV_NAME
        self.SEED = SEED

        self.MAX_MODEL = MAX_MODEL
        self.DEVICE=DEVICE

        self.ARCHITECTURE = ARCHITECTURE
        self.ACTIVATION = ACTIVATION
        self.SKIP = SKIP

        self.REWARD = REWARD
        self.PENALTY = PENALTY


class dqnConfig(config):
    def __init__(self,  ENV_NAME='Pendulum-v0',
                        DEVICE='cpu', SEED=0,
                        MAX_UPDATES=2000000, MAX_EP_STEPS=200,
                        EPSILON=0.95, EPS_END=0.05, EPS_PERIOD=1, EPS_DECAY=0.5,
                        EPS_RESET_PERIOD=100,
                        LR_C=1e-3, LR_C_END=1e-4, LR_C_PERIOD=1, LR_C_DECAY=0.5,
                        GAMMA=0.9, GAMMA_END=0.99999999, GAMMA_PERIOD=200, GAMMA_DECAY=0.5,
                        TAU=0.01, HARD_UPDATE=1, SOFT_UPDATE=True,
                        MEMORY_CAPACITY=10000,
                        BATCH_SIZE=32,
                        RENDER=False,
                        MAX_MODEL=10,
                        DOUBLE=True,
                        ARCHITECTURE=[512, 512, 512],
                        ACTIVATION='Tanh',
                        SKIP=False,
                        REWARD=-1,
                        PENALTY=1):
        """
        __init__

        Args:
            DOUBLE (bool, optional): use target network or not.
                Defaults to True.
            TAU (float, optional): soft update parameter of target network.
                Defaults to 0.01.
            HARD_UPDATE (int, optional): update period of target network if `SOFT_UPDATE` is False.
                Defaults to 1.
            SOFT_UPDATE (bool, optional): the way you update the target network.
                Defaults to True.
        """
        super().__init__(ENV_NAME=ENV_NAME,
                        DEVICE=DEVICE, SEED=SEED,
                        MAX_UPDATES=MAX_UPDATES, MAX_EP_STEPS=MAX_EP_STEPS,
                        EPSILON=EPSILON, EPS_END=EPS_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=EPS_DECAY,
                        EPS_RESET_PERIOD=EPS_RESET_PERIOD,
                        LR_C=LR_C, LR_C_END=LR_C_END, LR_C_PERIOD=LR_C_PERIOD, LR_C_DECAY=LR_C_DECAY,
                        GAMMA=GAMMA, GAMMA_END=GAMMA_END, GAMMA_PERIOD=GAMMA_PERIOD, GAMMA_DECAY=GAMMA_DECAY,
                        MEMORY_CAPACITY=MEMORY_CAPACITY,
                        BATCH_SIZE=BATCH_SIZE,
                        RENDER=RENDER,
                        MAX_MODEL=MAX_MODEL,
                        ARCHITECTURE=ARCHITECTURE,
                        ACTIVATION=ACTIVATION,
                        SKIP=SKIP,
                        REWARD=REWARD,
                        PENALTY=PENALTY)
        self.DOUBLE = DOUBLE
        self.TAU = TAU
        self.HARD_UPDATE = HARD_UPDATE
        self.SOFT_UPDATE = SOFT_UPDATE


class actorCriticConfig(config):
    def __init__(self,  ENV_NAME='Pendulum-v0', DEVICE='cpu', SEED=0,
        MAX_UPDATES=2000000, MAX_EP_STEPS=200,
        LR_C=1e-3, LR_C_END=1e-4, LR_C_PERIOD=1, LR_C_DECAY=0.5,
        LR_A=1e-3, LR_A_END=1e-4, LR_A_PERIOD=1, LR_A_DECAY=0.5,
        GAMMA=0.9, GAMMA_END=0.99999999, GAMMA_PERIOD=200, GAMMA_DECAY=0.5,
        TAU=0.05,
        MEMORY_CAPACITY=10000,
        BATCH_SIZE=64,
        MAX_MODEL=50,
        ARCHITECTURE=[512, 512, 512],
        ACTIVATION={'critic':'Sin', 'actor':'ReLU'},
        SKIP=False,
        REWARD=-1,
        PENALTY=1):
        """
        __init__

        Args:
            LR_A (float, optional): learning rate of actor model.
                Defaults to 1e-3.
            LR_A_END (float, optional): terminal value of LR_A.
                Defaults to 1e-4.
            LR_A_PERIOD (int, optional): update period of LR_A.
                Defaults to 1.
            LR_A_DECAY (float, optional): multiplicative factor of LR_A.
                Defaults to 0.5.
        """
        self.MAX_UPDATES = MAX_UPDATES
        self.MAX_EP_STEPS = MAX_EP_STEPS

        self.LR_C = LR_C
        self.LR_C_END = LR_C_END
        self.LR_C_PERIOD = LR_C_PERIOD
        self.LR_C_DECAY = LR_C_DECAY

        self.GAMMA = GAMMA
        self.GAMMA_END = GAMMA_END
        self.GAMMA_PERIOD = GAMMA_PERIOD
        self.GAMMA_DECAY = GAMMA_DECAY

        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE

        self.ENV_NAME = ENV_NAME
        self.SEED = SEED

        self.MAX_MODEL = MAX_MODEL
        self.DEVICE=DEVICE

        self.ARCHITECTURE = ARCHITECTURE
        self.ACTIVATION = ACTIVATION
        self.SKIP = SKIP

        self.REWARD = REWARD
        self.PENALTY = PENALTY

        self.LR_A = LR_A
        self.LR_A_END = LR_A_END
        self.LR_A_PERIOD = LR_A_PERIOD
        self.LR_A_DECAY = LR_A_DECAY

        self.TAU = TAU


class SACConfig(actorCriticConfig):
    def __init__(self,  ENV_NAME='Pendulum-v0', DEVICE='cpu', SEED=0,
        MAX_UPDATES=2000000, MAX_EP_STEPS=200,
        LR_C=1e-3, LR_C_END=1e-4, LR_C_PERIOD=1, LR_C_DECAY=0.5,
        LR_A=1e-3, LR_A_END=1e-4, LR_A_PERIOD=1, LR_A_DECAY=0.5,
        LR_Al=1e-4, LR_Al_END=1e-5, LR_Al_PERIOD=1, LR_Al_DECAY=0.5,
        GAMMA=0.9, GAMMA_END=0.99999999, GAMMA_PERIOD=200, GAMMA_DECAY=0.5,
        ALPHA=0.2, LEARN_ALPHA=True,
        TAU=0.05,
        MEMORY_CAPACITY=10000,
        BATCH_SIZE=64,
        MAX_MODEL=50,
        ARCHITECTURE=[512, 512, 512], ACTIVATION={'critic':'Sin', 'actor':'ReLU'},
        SKIP=False,
        REWARD=-1,
        PENALTY=1):

        super().__init__(ENV_NAME=ENV_NAME, DEVICE=DEVICE, SEED=SEED,
            MAX_UPDATES=MAX_UPDATES, MAX_EP_STEPS=MAX_EP_STEPS,
            LR_C=LR_C, LR_C_END=LR_C_END, LR_C_PERIOD=LR_C_PERIOD, LR_C_DECAY=LR_C_DECAY,
            LR_A=LR_A, LR_A_END=LR_A_END, LR_A_PERIOD=LR_A_PERIOD, LR_A_DECAY=LR_A_DECAY,
            GAMMA=GAMMA, GAMMA_END=GAMMA_END, GAMMA_PERIOD=GAMMA_PERIOD, GAMMA_DECAY=GAMMA_DECAY,
            TAU=TAU,
            MEMORY_CAPACITY=MEMORY_CAPACITY,
            BATCH_SIZE=BATCH_SIZE,
            MAX_MODEL=MAX_MODEL,
            ARCHITECTURE=ARCHITECTURE, ACTIVATION=ACTIVATION,
            SKIP=SKIP,
            REWARD=REWARD,
            PENALTY=PENALTY)

        self.LR_Al=LR_Al
        self.LR_Al_END=LR_Al_END
        self.LR_Al_PERIOD=LR_Al_PERIOD
        self.LR_Al_DECAY=LR_Al_DECAY
        self.ALPHA=ALPHA
        self.LEARN_ALPHA=LEARN_ALPHA