from gym.envs.registration import register


# Mujoco
# ----------------------------------------

register(
    'AntPos-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.ant:AntPosEnv'}
)

register(
    'Pusher-v0',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.pusher:PusherEnv'},
    max_episode_steps=100,  #* MAESN uses 100
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='maml_rl.envs.navigation:Navigation2DEnv',
    max_episode_steps=20	# was 100
)


