# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np

from utils.misc import Struct
from utils.color import rainbow
from utils.process_vanilla import generate_dataset

rng = np.random.default_rng(seed=0)

cfg_env = {
    'obs_type': ['circle', 'rect', 'rect'],
    'num_obs_range': [3, 5],
    'obs_loc_x_range': [0.6, 1.3],
    'obs_loc_y_range': [-0.7, 0.7],
    'obs_radius_range': [0.15, 0.3],
    'obs_width_range': [0.1, 0.4],
    'obs_height_range': [0.02, 0.05],
    'obs_theta_range': [-np.pi / 2, np.pi / 2],
    'obs_color': rainbow,
    'goal_loc_x_range': 1.6,
    'goal_loc_y_range': [-0.6, 0.6],
    'init_theta_range': [-np.pi / 3, np.pi / 3],
    'init_y_range': [-0.5, 0.5],
    # If `use_yaw_choices` is True, it selects the goal yaw range from a list
    # of tuples consisting of the lower and upper bound (`yaw_append_choices`).
    # Otherwise, it samples the lower bound from `yaw_append_lb_range` and adds
    # `yaw_append_range` to get the upper bound.
    'use_yaw_choices': True,
    'yaw_append_choices': np.array([[0., 2 * np.pi]]),
    'yaw_append_range': [0, 0],
    'bounds': np.array([[0., 2], [-1, 1], [0, 2 * np.pi]]),
    'get_append': False,
    'goal_dist_thr': 0.2,
    'init_dist_thr': 0.2,
}
cfg_env = Struct(**cfg_env)

subfolder = 'vanilla'
generate_dataset(subfolder, cfg_env, rng)
