from gym.envs.registration import register


# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='maml_rl.envs.navigation:Navigation2DEnv',
    max_episode_steps=20	# was 100
)

# 2D Navigation
# ----------------------------------------

register(
    'NavigationObs-v0',
    entry_point='maml_rl.envs.navigation_obs:NavigationObsEnv',
    max_episode_steps=50
)

register(
    'NavigationPBObs-v0',
    entry_point='maml_rl.envs.navigation_obs:NavigationObsPBEnv',
    max_episode_steps=50
)
