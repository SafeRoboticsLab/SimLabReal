# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Split tasks for Advanced-Realistic setting

Load all tasks generated for Advanced-Realistic setting and split them into train and test sets for posterior policy training.

Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
         Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import argparse
import random
import glob
import pickle

from utils.misc import save_obj


def main(args):
    # count number of tasks available
    path_all = glob.glob(args.task_folder + '*/*/task.pkl')
    print('Number of tasks available: {}'.format(len(path_all)))
    assert len(path_all) > 2500

    # Aggregate tasks
    save_tasks = []
    for path in path_all:
        with open(path, 'rb') as f:
            task = pickle.load(f)[0]
        save_tasks += [task]

    # Shuffle tasks
    random.shuffle(save_tasks)

    # Save prior, validation, posterior, test set
    save_obj(
        save_tasks[0:500],
        os.path.join(args.save_folder, 'advanced_realistic_ps_train_500')
    )
    save_obj(
        save_tasks[500:1500],
        os.path.join(args.save_folder, 'advanced_realistic_ps_train_1000')
    )
    save_obj(
        save_tasks[1500:2500],
        os.path.join(args.save_folder, 'advanced_realistic_ps_test_1000')
    )
    save_obj(
        save_tasks[500:2500],
        os.path.join(args.save_folder, 'advanced_realistic_ps_combine_2000')
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_folder', default='/home/allen/data/pacra/', nargs='?',
        help='path to save the task datasets'
    )
    parser.add_argument(
        '--task_folder',
        default='/home/allen/data/pacra/3d-front-tasks-advanced-realistic/',
        nargs='?', help='path to all tasks of the Advanced Realistic setting'
    )
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
