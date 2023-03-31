# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# usage: python gen_lab_dyn.py -th -thl -0.7 -0.3
# usage: python gen_lab_dyn.py -p -xpl 0.3 0.5 -xpu 0.8 1.0

from typing import Union, Optional
import os
import pickle
import copy
import argparse
import numpy as np

from utils.misc import sample_with_range
from utils.misc import save_obj


def get_samples(
    data_range: Union[np.ndarray, list, float], n_samples: int,
    rng: np.random.Generator
) -> np.ndarray:
    """Gets data by sampling from the range.

    Args:
        data_range (Union[np.ndarray, list, float]): the range for sampling.
        n_samples (int): the number of samples.
        rng (np.random.Generator): random number generator.

    Returns:
        np.ndarray (n_samples, 1): sampled values.
    """
    if isinstance(data_range, np.ndarray) or isinstance(data_range, list):
        if len(data_range) > 1:
            if data_range[0] == data_range[1]:
                data = np.ones(n_samples) * data_range[0]
            else:
                data = sample_with_range(data_range, n_samples, rng)
        else:
            data = np.ones(n_samples) * data_range
    else:
        data = np.ones(n_samples) * data_range
    return data.reshape(-1, 1)


def sample_lb_ub_with_range(
    lb_range: Union[np.ndarray, list, float], ub_range: Union[np.ndarray, list,
                                                              float],
    n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Samples lower and upper bounds from the range.

    Args:
        lb_range (Union[np.ndarray, list, float]): the lower bound range for
            sampling.
        ub_range (Union[np.ndarray, list, float]): the upper bound range for
            sampling.
        n_samples (int): the number of samples.
        rng (np.random.Generator): random number generator.

    Returns:
        np.ndarray (n_samples, 2): sampled lower and upper bounds.
    """
    lb = get_samples(lb_range, n_samples, rng)
    ub = get_samples(ub_range, n_samples, rng)
    return np.concatenate((lb, ub), axis=1)


def change_dyn(
    src_folder: str, filename: str, rng: np.random.Generator,
    change_xdot_perf: bool, change_xdot_backup: bool, change_thetadot: bool,
    xdot_perf_lb_range: Optional[Union[np.ndarray, list, float]] = None,
    xdot_perf_ub_range: Optional[Union[np.ndarray, list, float]] = None,
    xdot_backup_lb_range: Optional[Union[np.ndarray, list, float]] = None,
    xdot_backup_ub_range: Optional[Union[np.ndarray, list, float]] = None,
    thetadot_lb_range: Optional[Union[np.ndarray, list, float]] = None,
    thetadot_ub_range: Optional[Union[np.ndarray, list, float]] = None
):
    data_path = os.path.join('data', src_folder, filename + '.pkl')
    with open(data_path, 'rb') as f:
        src_dataset = pickle.load(f)
    num_tasks = len(src_dataset)
    tar_dataset = copy.deepcopy(src_dataset)

    if change_xdot_perf:
        if xdot_perf_lb_range is None:
            xdot_perf_lb_range = src_dataset[0]['xdot_range_perf'][0]
        if xdot_perf_ub_range is None:
            xdot_perf_ub_range = src_dataset[0]['xdot_range_perf'][1]
        xdot_perf_range_all = sample_lb_ub_with_range(
            xdot_perf_lb_range, xdot_perf_ub_range, num_tasks, rng
        )

    if change_xdot_backup:
        if xdot_backup_lb_range is None:
            xdot_backup_lb_range = src_dataset[0]['xdot_range_backup'][0]
        if xdot_backup_ub_range is None:
            xdot_backup_ub_range = src_dataset[0]['xdot_range_backup'][1]
        xdot_backup_range_all = sample_lb_ub_with_range(
            xdot_backup_lb_range, xdot_backup_ub_range, num_tasks, rng
        )

    if change_thetadot:
        if thetadot_lb_range is None:
            thetadot_lb_range = -1.
        if thetadot_ub_range is None:
            thetadot_ub_range = 1.
        thetadot_range_all = sample_lb_ub_with_range(
            thetadot_lb_range, thetadot_ub_range, num_tasks, rng
        )

    for i, task in enumerate(tar_dataset):
        if change_xdot_perf:
            task['xdot_range_perf'] = xdot_perf_range_all[i]
        if change_xdot_backup:
            task['xdot_range_backup'] = xdot_backup_range_all[i]
        if change_thetadot:
            task['thetadot_range'] = thetadot_range_all[i]
        else:
            task['thetadot_range'] = np.array([-1., 1.])

    return tar_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--change_xdot_perf", help="change xdot perf",
        action='store_true'
    )
    parser.add_argument(
        "-b", "--change_xdot_backup", help="change xdot backup",
        action='store_true'
    )
    parser.add_argument(
        "-th", "--change_thetadot", help="change thetadot", action='store_true'
    )

    parser.add_argument(
        "-xpl", "--xdot_perf_lb_range", help="xdot perf LB range",
        default=None, nargs="*", type=float
    )
    parser.add_argument(
        "-xpu", "--xdot_perf_ub_range", help="xdot perf UB range",
        default=None, nargs="*", type=float
    )
    parser.add_argument(
        "-xbl", "--xdot_backup_lb_range", help="xdot backup LB range",
        default=None, nargs="*", type=float
    )
    parser.add_argument(
        "-xbu", "--xdot_backup_ub_range", help="xdot backup UB range",
        default=None, nargs="*", type=float
    )
    parser.add_argument(
        "-thl", "--thetadot_lb_range", help="thetadot LB range", default=None,
        nargs="*", type=float
    )
    parser.add_argument(
        "-thu", "--thetadot_ub_range", help="thetadot UB range", default=None,
        nargs="*", type=float
    )
    parser.add_argument(
        "-sf", "--src_folder", help="source folder", type=str, default='toy_v2'
    )
    parser.add_argument(
        "-tf", "--tar_folder", help="target folder", type=str,
        default='toy_v2_task_dyn'
    )

    args = parser.parse_args()
    rng = np.random.default_rng(seed=0)

    src_folder = args.src_folder
    tar_folder = args.tar_folder
    data_folder = data_folder = os.path.join('data', tar_folder)
    os.makedirs(data_folder, exist_ok=True)

    filename = 'train_posterior_2000'
    tar_dataset = change_dyn(
        src_folder, filename, rng, args.change_xdot_perf,
        args.change_xdot_backup, args.change_thetadot,
        xdot_perf_lb_range=args.xdot_perf_lb_range,
        xdot_perf_ub_range=args.xdot_perf_ub_range,
        xdot_backup_lb_range=args.xdot_backup_lb_range,
        xdot_backup_ub_range=args.xdot_backup_ub_range,
        thetadot_lb_range=args.thetadot_lb_range,
        thetadot_ub_range=args.thetadot_ub_range
    )

    for i in range(10):
        task = tar_dataset[i]
        print("==", i, "==")
        print(task['xdot_range_perf'])
        print(task['xdot_range_backup'])
        print(task['thetadot_range'])

    train_ps_1000 = rng.choice(tar_dataset, size=1000, replace=False)
    train_ps_500 = rng.choice(train_ps_1000, size=500, replace=False)
    train_ps_100 = rng.choice(train_ps_500, size=100, replace=False)
    save_obj(tar_dataset, os.path.join(data_folder, 'train_posterior_2000'))
    save_obj(train_ps_100, os.path.join(data_folder, 'train_posterior_100'))
    save_obj(train_ps_500, os.path.join(data_folder, 'train_posterior_500'))
    save_obj(train_ps_1000, os.path.join(data_folder, 'train_posterior_1000'))

    filename = 'test'
    tar_dataset = change_dyn(
        src_folder, filename, rng, args.change_xdot_perf,
        args.change_xdot_backup, args.change_thetadot,
        xdot_perf_lb_range=args.xdot_perf_lb_range,
        xdot_perf_ub_range=args.xdot_perf_ub_range,
        xdot_backup_lb_range=args.xdot_backup_lb_range,
        xdot_backup_ub_range=args.xdot_backup_ub_range,
        thetadot_lb_range=args.thetadot_lb_range,
        thetadot_ub_range=args.thetadot_ub_range
    )
    save_obj(tar_dataset, os.path.join(data_folder, 'test'))
