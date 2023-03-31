# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import pickle
import copy
import numpy as np
import argparse

os.sys.path.append(os.path.join(os.getcwd(), '.'))

from utils.misc import save_obj, trans2yaw_range


def change_task(
    src_folder, filename, yaw_append=None, use_sample=False, rng=None,
    yaw_append_choices=None
):
    data_path = os.path.join('data', src_folder, filename + '.pkl')
    with open(data_path, 'rb') as f:
        src_dataset = pickle.load(f)
    tar_dataset = copy.deepcopy(src_dataset)

    if use_sample:
        yaw_indices = rng.integers(
            low=0, high=len(yaw_append_choices), size=len(tar_dataset)
        )
    else:
        yaw_range = trans2yaw_range(yaw_append)

    for i, task in enumerate(tar_dataset):
        if use_sample:
            yaw_index = yaw_indices[i]
            yaw_append = yaw_append_choices[yaw_index]
            yaw_range = trans2yaw_range(yaw_append)

        task['goal_yaw_range'] = yaw_range
        if task['get_yaw_append']:
            task['yaw_append'] = yaw_append

    return tar_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--src_folder", help="source folder", type=str)
    parser.add_argument("-tf", "--tar_folder", help="target folder", type=str)
    parser.add_argument("-up", "--update_prior", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=0)
    yaw_append_low = np.arange(-90, 30.1, 15).reshape(-1, 1) * np.pi / 180
    yaw_append_high = yaw_append_low + np.pi / 3
    yaw_append_choices = np.concatenate([yaw_append_low, yaw_append_high],
                                        axis=1)
    yaw_append_choices[np.abs(yaw_append_choices) < 1e-8] = 0.0

    src_folder = args.src_folder
    tar_folder = args.tar_folder
    data_folder = os.path.join('data', tar_folder)
    os.makedirs(data_folder, exist_ok=True)

    if args.update_prior:
        filename = 'train_prior_1000'
        tar_dataset = change_task(
            src_folder, filename, use_sample=True, rng=rng,
            yaw_append_choices=yaw_append_choices
        )
        for i in range(10):
            task = tar_dataset[i]
            print("==", i, "==")
            print(task['goal_yaw_range'] * 180 / np.pi)
            print(task['yaw_append'] * 180 / np.pi)

        train_prior_100 = rng.choice(tar_dataset, size=100, replace=False)
        train_prior_10 = rng.choice(train_prior_100, size=10, replace=False)
        save_obj(tar_dataset, os.path.join(data_folder, 'train_prior_1000'))
        save_obj(train_prior_100, os.path.join(data_folder, 'train_prior_100'))
        save_obj(train_prior_10, os.path.join(data_folder, 'train_prior_10'))

    # Fix the lab and real to be a single range.
    yaw_append = np.array([-np.pi / 3, 0.])

    filename = 'train_posterior_2000'
    tar_dataset = change_task(src_folder, filename, yaw_append=yaw_append)
    for i in range(10):
        task = tar_dataset[i]
        print("==", i, "==")
        print(task['goal_yaw_range'] * 180 / np.pi)
        if task['get_yaw_append']:
            print(task['yaw_append'] * 180 / np.pi)

    train_ps_1000 = rng.choice(tar_dataset, size=1000, replace=False)
    train_ps_500 = rng.choice(train_ps_1000, size=500, replace=False)
    train_ps_100 = rng.choice(train_ps_500, size=100, replace=False)
    save_obj(tar_dataset, os.path.join(data_folder, 'train_posterior_2000'))
    save_obj(train_ps_100, os.path.join(data_folder, 'train_posterior_100'))
    save_obj(train_ps_500, os.path.join(data_folder, 'train_posterior_500'))
    save_obj(train_ps_1000, os.path.join(data_folder, 'train_posterior_1000'))

    filename = 'test'
    tar_dataset = change_task(src_folder, filename, yaw_append=yaw_append)
    save_obj(tar_dataset, os.path.join(data_folder, 'test'))
