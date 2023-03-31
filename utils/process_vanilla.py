# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Utils for generating datasets for vanilla environments.
"""

import os
import pickle
import copy
import numpy as np
from utils.misc import (
    get_feasible_rect, get_feasible_circle, get_occupancy_map, trans2yaw_range,
    sample_with_range, save_obj
)


def generate_dataset(
    subfolder: str, cfg_env, rng: np.random.Generator, get_prior: bool = True
):
    data_folder = os.path.join('data', subfolder)
    os.makedirs(data_folder, exist_ok=True)

    if get_prior:
        print("\nValidation")
        validation = sample_toy_tasks(
            cfg_env, n_task=100, decorator='validation', rng=rng
        )
        save_obj(validation, os.path.join(data_folder, 'validation'))
        for i in range(10):
            task = validation[i]
            print("==", i, "==")
            print(task['goal_yaw_range'] * 180 / np.pi)
            if cfg_env.get_append:
                if cfg_env.use_yaw_choices:
                    if cfg_env.use_append_one_hot:
                        print(task['yaw_append'])
                    else:
                        print(task['yaw_append'] * 180 / np.pi)
                else:
                    print(task['yaw_append'] * 180 / np.pi)

        print("Prior")
        train_prior = sample_toy_tasks(
            cfg_env, n_task=1000, decorator='train_prior', rng=rng
        )
        train_prior_100 = rng.choice(train_prior, size=100, replace=False)
        train_prior_10 = rng.choice(train_prior_100, size=10, replace=False)
        save_obj(train_prior, os.path.join(data_folder, 'train_prior_1000'))
        save_obj(train_prior_10, os.path.join(data_folder, 'train_prior_10'))
        save_obj(train_prior_100, os.path.join(data_folder, 'train_prior_100'))

    print("\nPosterior")
    train_ps = sample_toy_tasks(
        cfg_env, n_task=2000, decorator='train_posterior', rng=rng
    )
    train_ps_1000 = rng.choice(train_ps, size=1000, replace=False)
    train_ps_500 = rng.choice(train_ps_1000, size=500, replace=False)
    train_ps_100 = rng.choice(train_ps_500, size=100, replace=False)
    save_obj(train_ps, os.path.join(data_folder, 'train_posterior_2000'))
    save_obj(train_ps_100, os.path.join(data_folder, 'train_posterior_100'))
    save_obj(train_ps_500, os.path.join(data_folder, 'train_posterior_500'))
    save_obj(train_ps_1000, os.path.join(data_folder, 'train_posterior_1000'))
    for i in range(10):
        task = train_ps_100[i]
        print("==", i, "==")
        print(task['goal_yaw_range'] * 180 / np.pi)
        if cfg_env.get_append:
            if cfg_env.use_yaw_choices:
                if cfg_env.use_append_one_hot:
                    print(task['yaw_append'])
                else:
                    print(task['yaw_append'] * 180 / np.pi)
            else:
                print(task['yaw_append'] * 180 / np.pi)

    print("\nTest")
    test = sample_toy_tasks(cfg_env, n_task=2000, decorator='test', rng=rng)
    save_obj(test, os.path.join(data_folder, 'test'))


def sample_toy_tasks(cfg_env, n_task, decorator, rng):
    task_all = []
    bounds = cfg_env.bounds
    use_yaw_choices = cfg_env.use_yaw_choices

    init_theta_all = sample_with_range(cfg_env.init_theta_range, n_task, rng)
    init_y_all = sample_with_range(cfg_env.init_y_range, n_task, rng)
    goal_y_all = sample_with_range(cfg_env.goal_loc_y_range, n_task, rng)
    num_obs_all = rng.integers(
        cfg_env.num_obs_range[0], cfg_env.num_obs_range[1], endpoint=True,
        size=n_task
    )
    if use_yaw_choices:
        if cfg_env.yaw_append_range[0] == cfg_env.yaw_append_range[1]:
            yaw_append_indices_all = np.ones(shape=(n_task,), dtype=int
                                            ) * cfg_env.yaw_append_range[0]
        else:
            yaw_append_indices_all = rng.integers(
                low=cfg_env.yaw_append_range[0],
                high=cfg_env.yaw_append_range[1], endpoint=True, size=n_task
            )
    else:
        yaw_append_lb_all = sample_with_range(
            cfg_env.yaw_append_lb_range, n_task, rng
        )

    # Constructs each task given the pre-sampled values.
    for task_id in range(n_task):
        print(task_id, end='\r')
        task = {}
        task['id'] = decorator + '_' + str(task_id)

        # Samples the initial position.
        task['init_state'] = np.array([
            0.1, init_y_all[task_id], init_theta_all[task_id]
        ])

        # Samples the goal position.
        task['goal_loc'] = np.array([
            cfg_env.goal_loc_x_range, goal_y_all[task_id]
        ])
        task['goal_radius'] = 0.15

        # Samples obstacles.
        num_obs = num_obs_all[task_id]
        obs_dict = []
        while len(obs_dict) < num_obs:
            tmp_d, cross_d = sample_obstacle(cfg_env, task, rng)

            if tmp_d:
                obs_dict.append(tmp_d)

            if cross_d:
                obs_dict.append(cross_d)

        task['num_obs'] = len(obs_dict)
        task['obs_dict'] = obs_dict

        # map
        task['grid_cells'] = (100, 100)
        task['occupancy_map'] = get_occupancy_map(bounds, task)

        # dynamics
        task['xdot_range_perf'] = [0.2, 1.]
        task['xdot_range_backup'] = [0.2, 0.5]

        # goal yaw range
        if use_yaw_choices:
            # Selects the goal yaw range from a list of tuples consisting of
            # the lower and upper bound (`yaw_append_choices`).
            yaw_index = yaw_append_indices_all[task_id]
            yaw_append_one_hot = np.zeros(len(cfg_env.yaw_append_choices))
            yaw_append_one_hot[yaw_index] = 1
            yaw_append = cfg_env.yaw_append_choices[yaw_index]
        else:
            # Gets lower bound from the sampling before the for loop and
            # adds the range to get the upper bound.
            yaw_append_lb = yaw_append_lb_all[task_id]
            yaw_append_range = cfg_env.yaw_append_range
            yaw_append = np.array([
                yaw_append_lb, yaw_append_lb + yaw_append_range
            ])
        yaw_range = trans2yaw_range(yaw_append)
        task['goal_yaw_range'] = yaw_range

        # additional append signal in step info
        task['get_yaw_append'] = cfg_env.get_append
        if cfg_env.get_append:
            if use_yaw_choices:
                if cfg_env.use_append_one_hot:
                    task['yaw_append'] = yaw_append_one_hot
                else:
                    task['yaw_append'] = yaw_append
            else:
                task['yaw_append'] = yaw_append

        task_all += [task]

    return task_all


def sample_obstacle(config, task, rng):
    bounds = config.bounds
    goal_dist_thr = config.goal_dist_thr
    init_dist_thr = config.init_dist_thr

    obs_type = rng.integers(3, size=1)[0]
    obs_x = sample_with_range(config.obs_loc_x_range, 1, rng)[0]
    obs_y = sample_with_range(config.obs_loc_y_range, 1, rng)[0]
    obs_rgba = rng.integers(7, size=1)[0]

    tmp_d = {}
    tmp_d['obs_type'] = config.obs_type[obs_type]
    tmp_d['rgba'] = config.obs_color[obs_rgba]
    tmp_d['loc'] = np.array([obs_x, obs_y])
    cross_d = None
    if obs_type == 0:
        tmp_radius = sample_with_range(config.obs_radius_range, 1, rng)[0]

        # Make sure the obstacle is not too close to the goal.
        tmp_radius = get_feasible_circle(
            tmp_d['loc'], tmp_radius, task['goal_loc'], task['goal_radius'],
            goal_dist_thr, config.obs_radius_range[0]
        )
        if tmp_radius is None:
            return None, None

        # Make sure the obstacle is not too close to the init.
        tmp_radius = get_feasible_circle(
            tmp_d['loc'], tmp_radius, task['init_state'][:2], 0.,
            init_dist_thr, config.obs_radius_range[0]
        )
        if tmp_radius is not None:
            tmp_d['radius'] = tmp_radius
        else:
            return None, None
    else:
        tmp_width = sample_with_range(config.obs_width_range, 1, rng)[0]
        tmp_d['height'] = sample_with_range(config.obs_height_range, 1, rng)[0]
        tmp_d['theta'] = sample_with_range(config.obs_theta_range, 1, rng)[0]

        # Make sure the obstacle is not too close to the goal.
        tmp_width = get_feasible_rect(
            tmp_d['loc'], tmp_width, tmp_d['height'], tmp_d['theta'],
            task['goal_loc'], task['goal_radius'], goal_dist_thr,
            step_size=0.01, min_width=0.1
        )
        if tmp_width is None:
            return None, None

        # Make sure the obstacle is not too close to the init.
        tmp_width = get_feasible_rect(
            tmp_d['loc'], tmp_width, tmp_d['height'], tmp_d['theta'],
            task['init_state'][:2], 0., init_dist_thr, step_size=0.01,
            min_width=0.1
        )
        if tmp_width is not None:
            tmp_d['width'] = tmp_width
        else:
            return None, None

        # Get a cross obstacle (counted as two obstacles).
        if obs_type == 2:
            half_ell = tmp_d['width']
            cross_pt_dist = sample_with_range([-half_ell, half_ell], 1, rng)[0]
            cross_pt = [
                obs_x + cross_pt_dist * np.cos(tmp_d['theta']),
                obs_y + cross_pt_dist * np.sin(tmp_d['theta'])
            ]

            cross_d = {}
            tmp_width = sample_with_range(config.obs_width_range, 1, rng)[0]
            cross_d['height'] = sample_with_range(
                config.obs_height_range, 1, rng
            )[0]
            cross_d['obs_type'] = 'rect'
            cross_d['rgba'] = config.obs_color[rng.integers(7, size=1)[0]]

            if tmp_d['theta'] > 0:
                cross_d['theta'] = -(np.pi / 2 - tmp_d['theta'])
            else:
                cross_d['theta'] = np.pi / 2 + tmp_d['theta']

            cross_center_dist = sample_with_range([-tmp_width, tmp_width], 1,
                                                  rng)[0]
            tmp_cross_x = cross_pt[0] + cross_center_dist * np.sin(
                tmp_d['theta']
            )
            tmp_cross_y = cross_pt[1] + cross_center_dist * np.cos(
                tmp_d['theta']
            )
            tmp_cross_x = max(tmp_cross_x, bounds[0][0])
            tmp_cross_x = min(tmp_cross_x, bounds[0][1])
            tmp_cross_y = max(tmp_cross_y, bounds[1][0])
            tmp_cross_y = min(tmp_cross_y, bounds[1][1])

            cross_d['loc'] = np.array([tmp_cross_x, tmp_cross_y])

            # Make sure the obstacle is not too close to the goal.
            tmp_width = get_feasible_rect(
                cross_d['loc'], tmp_width, cross_d['height'], cross_d['theta'],
                task['goal_loc'], task['goal_radius'], goal_dist_thr,
                step_size=0.01, min_width=config.obs_width_range[0]
            )
            if tmp_width is None:
                return None, None

            # Make sure the obstacle is not too close to the init.
            tmp_width = get_feasible_rect(
                cross_d['loc'], tmp_width, cross_d['height'], cross_d['theta'],
                task['init_state'][:2], 0., init_dist_thr, step_size=0.01,
                min_width=config.obs_width_range[0]
            )
            if tmp_width is not None:
                cross_d['width'] = tmp_width
            else:
                return None, None

    return tmp_d, cross_d


def change_task_yaw_by_choices(
    src_folder, filename, rng, yaw_append_choices, use_sample=False,
    yaw_index=None, yaw_append_idx_range=None
):
    if use_sample:
        assert yaw_append_idx_range is not None, \
            "Yaw append range is required if using sampling"
    else:
        assert yaw_index is not None, \
            "Yaw index is required if not using sampling"
    data_path = os.path.join('data', src_folder, filename + '.pkl')
    with open(data_path, 'rb') as f:
        src_dataset = pickle.load(f)
    tar_dataset = copy.deepcopy(src_dataset)

    if use_sample:
        yaw_indices = rng.integers(
            low=yaw_append_idx_range[0], high=yaw_append_idx_range[1],
            size=len(tar_dataset)
        )
    for i, task in enumerate(tar_dataset):
        if use_sample:
            yaw_index = yaw_indices[i]
        yaw_append_one_hot = np.zeros(len(yaw_append_choices))
        yaw_append_one_hot[yaw_index] = 1
        yaw_append = yaw_append_choices[yaw_index]
        yaw_range = trans2yaw_range(yaw_append)

        task['goal_yaw_range'] = yaw_range
        task['yaw_append'] = yaw_append_one_hot
        task['get_yaw_append'] = True

    return tar_dataset


def change_task_yaw_by_lb(
    src_folder, filename, rng, yaw_append_range, use_sample=False, yaw_lb=None,
    yaw_append_lb_range=None
):
    if use_sample:
        assert yaw_append_lb_range is not None, \
            "Yaw append lb range is required if using sampling"
    else:
        assert yaw_lb is not None, \
            "Yaw lb is required if not using sampling"
    data_path = os.path.join('data', src_folder, filename + '.pkl')
    with open(data_path, 'rb') as f:
        src_dataset = pickle.load(f)
    tar_dataset = copy.deepcopy(src_dataset)

    if use_sample:
        yaw_append_lb_all = sample_with_range(
            yaw_append_lb_range, len(tar_dataset), rng
        )
    for i, task in enumerate(tar_dataset):
        if use_sample:
            yaw_append_lb = yaw_append_lb_all[i]
        yaw_append = np.array([
            yaw_append_lb, yaw_append_lb + yaw_append_range
        ])
        yaw_range = trans2yaw_range(yaw_append)

        task['goal_yaw_range'] = yaw_range
        task['yaw_append'] = yaw_append
        task['get_yaw_append'] = True

    return tar_dataset


def generate_lab_dataset_by_choices(
    src_folder, tar_folder, rng, yaw_append_choices, use_sample=False,
    yaw_index=None, yaw_append_idx_range=None, update_prior=False
):
    data_folder = data_folder = os.path.join('data', tar_folder)
    os.makedirs(data_folder, exist_ok=True)

    if update_prior:
        filename = 'train_prior_1000'
        tar_dataset = change_task_yaw_by_choices(
            src_folder, filename, rng=rng,
            yaw_append_choices=yaw_append_choices, use_sample=use_sample,
            yaw_append_idx_range=yaw_append_idx_range, yaw_index=yaw_index
        )
        for i in range(10):
            task = tar_dataset[i]
            print("==", i, "==")
            print("yaw range", task['goal_yaw_range'] * 180 / np.pi)
            print("yaw append", task['yaw_append'])

        train_prior_100 = rng.choice(tar_dataset, size=100, replace=False)
        train_prior_10 = rng.choice(train_prior_100, size=10, replace=False)
        save_obj(tar_dataset, os.path.join(data_folder, 'train_prior_1000'))
        save_obj(train_prior_100, os.path.join(data_folder, 'train_prior_100'))
        save_obj(train_prior_10, os.path.join(data_folder, 'train_prior_10'))

    filename = 'train_posterior_2000'
    tar_dataset = change_task_yaw_by_choices(
        src_folder, filename, rng=rng, yaw_append_choices=yaw_append_choices,
        use_sample=use_sample, yaw_append_idx_range=yaw_append_idx_range,
        yaw_index=yaw_index
    )
    for i in range(10):
        task = tar_dataset[i]
        print("==", i, "==")
        print("yaw range", task['goal_yaw_range'] * 180 / np.pi)
        print("yaw append", task['yaw_append'])

    train_ps_1000 = rng.choice(tar_dataset, size=1000, replace=False)
    train_ps_500 = rng.choice(train_ps_1000, size=500, replace=False)
    train_ps_100 = rng.choice(train_ps_500, size=100, replace=False)
    save_obj(tar_dataset, os.path.join(data_folder, 'train_posterior_2000'))
    save_obj(train_ps_100, os.path.join(data_folder, 'train_posterior_100'))
    save_obj(train_ps_500, os.path.join(data_folder, 'train_posterior_500'))
    save_obj(train_ps_1000, os.path.join(data_folder, 'train_posterior_1000'))

    filename = 'test'
    tar_dataset = change_task_yaw_by_choices(
        src_folder, filename, rng=rng, yaw_append_choices=yaw_append_choices,
        use_sample=use_sample, yaw_append_idx_range=yaw_append_idx_range,
        yaw_index=yaw_index
    )
    save_obj(tar_dataset, os.path.join(data_folder, 'test'))


def generate_lab_dataset_by_lb(
    src_folder, tar_folder, rng, yaw_append_range, use_sample=False,
    yaw_lb=None, yaw_append_lb_range=None, update_prior=False
):
    data_folder = data_folder = os.path.join('data', tar_folder)
    os.makedirs(data_folder, exist_ok=True)

    if update_prior:
        filename = 'train_prior_1000'
        tar_dataset = change_task_yaw_by_lb(
            src_folder, filename, rng=rng, yaw_append_range=yaw_append_range,
            use_sample=use_sample, yaw_append_lb_range=yaw_append_lb_range,
            yaw_lb=yaw_lb
        )
        for i in range(10):
            task = tar_dataset[i]
            print("==", i, "==")
            print("yaw range", task['goal_yaw_range'] * 180 / np.pi)
            print("yaw append", task['yaw_append'] * 180 / np.pi)

        train_prior_100 = rng.choice(tar_dataset, size=100, replace=False)
        train_prior_10 = rng.choice(train_prior_100, size=10, replace=False)
        save_obj(tar_dataset, os.path.join(data_folder, 'train_prior_1000'))
        save_obj(train_prior_100, os.path.join(data_folder, 'train_prior_100'))
        save_obj(train_prior_10, os.path.join(data_folder, 'train_prior_10'))

    filename = 'train_posterior_2000'
    tar_dataset = change_task_yaw_by_lb(
        src_folder, filename, rng=rng, yaw_append_range=yaw_append_range,
        use_sample=use_sample, yaw_append_lb_range=yaw_append_lb_range,
        yaw_lb=yaw_lb
    )
    for i in range(10):
        task = tar_dataset[i]
        print("==", i, "==")
        print("yaw range", task['goal_yaw_range'] * 180 / np.pi)
        print("yaw append", task['yaw_append'] * 180 / np.pi)

    train_ps_1000 = rng.choice(tar_dataset, size=1000, replace=False)
    train_ps_500 = rng.choice(train_ps_1000, size=500, replace=False)
    train_ps_100 = rng.choice(train_ps_500, size=100, replace=False)
    save_obj(tar_dataset, os.path.join(data_folder, 'train_posterior_2000'))
    save_obj(train_ps_100, os.path.join(data_folder, 'train_posterior_100'))
    save_obj(train_ps_500, os.path.join(data_folder, 'train_posterior_500'))
    save_obj(train_ps_1000, os.path.join(data_folder, 'train_posterior_1000'))

    filename = 'test'
    tar_dataset = change_task_yaw_by_lb(
        src_folder, filename, rng=rng, yaw_append_range=yaw_append_range,
        use_sample=use_sample, yaw_append_lb_range=yaw_append_lb_range,
        yaw_lb=yaw_lb
    )
    save_obj(tar_dataset, os.path.join(data_folder, 'test'))
