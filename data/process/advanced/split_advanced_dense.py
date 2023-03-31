# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Split tasks for Advanced-Dense setting

Load all tasks generated for Advanced-Dense setting and split them into train and test sets for prior policy training.

Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
         Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import argparse
import pickle
import random

from utils.misc import save_obj


def main(args):
    # count number of non-empty folders - sometimes we generate empty folders that the task generation was aborted
    path_all = []
    for path in os.listdir(args.task_folder):
        if len(os.listdir(os.path.join(args.task_folder, path))) > 0:
            path_all += [path]
    num_tasks_available = len(path_all)
    assert num_tasks_available > 2200

    # Aggregate tasks
    save_tasks = []
    task_ind = 0
    for path in path_all:
        task_path = os.path.join(args.task_folder, path, 'task.pkl')
        with open(task_path, 'rb') as f:
            task = pickle.load(f)[0]

        # Modify task id
        # task['task_id'] = task_ind

        # Save task
        save_tasks += [task]
        # task_ind += 1

    # Shuffle tasks
    random.shuffle(save_tasks)

    # Save
    save_obj(
        save_tasks[0:500],
        os.path.join(args.save_folder, 'advanced_dense_prior_train_500')
    )
    save_obj(
        save_tasks[500:1500],
        os.path.join(args.save_folder, 'advanced_dense_ps_train_1000')
    )
    save_obj(
        save_tasks[1500:2200],
        os.path.join(args.save_folder, 'advanced_dense_test_700')
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_folder', default='/home/allen/data/pacra', nargs='?',
        help='path to save the task datasets'
    )
    parser.add_argument(
        '--task_folder',
        default='/home/allen/data/pacra/3d-front-tasks-advanced-dense',
        nargs='?', help='path to all tasks of the Advanced Dense setting'
    )
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
