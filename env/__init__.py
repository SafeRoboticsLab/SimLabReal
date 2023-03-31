from env.vanilla_env import VanillaEnv
from env.advanced_env import AdvancedEnv
from env.advanced_dense_env import AdvancedDenseEnv

env_dict = {
    'vanilla': VanillaEnv,
    'advanced-realistic': AdvancedEnv,
    'advanced-dense': AdvancedDenseEnv,
}
