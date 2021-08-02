import gym
from safe_rl.navigation_obs_pb_cont import NavigationObsPBEnvCont

gym.envs.register(  # no time limit imposed
    id='NavigationObsPBEnvCont-v0',
    entry_point='safe_rl.navigation_obs_pb_cont:NavigationObsPBEnvCont',
)
