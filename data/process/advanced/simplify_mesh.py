# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Simplifying the OBJ meshes from the 3D Future dataset.

This file implements the simplification of the OBJ meshes from the 3D Future 
dataset using Blender. See blender_simplify.py for the details of the
simplification. 

Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
         Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import argparse
import glob
import shutil
import subprocess
import time
import concurrent.futures


def process(full_path):
    """Each folder is for one piece of furniture"""
    mesh_files = glob.glob(full_path + '/*raw_model.obj')
    for mesh in mesh_files:
        new_mesh = mesh.split('.')[0] + '_simplified.obj'
        bash_command = "blender -b -P data/process/advanced/blender_simplify.py -- --ratio 0.2 --inm " + mesh + " --outm " + new_mesh
        subprocess.check_output(['bash', '-c', bash_command])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mesh_folder', default='/home/temp/3d-front/3D-FUTURE-model/',
        nargs='?', help='path to load the meshes'
    )
    parser.add_argument(
        '--num_cpus',
        default=16,
        nargs='?',
        help='number of cpus to use',
    )
    args = parser.parse_args()

    # Run in parallel
    job_args = []
    folders_all = os.listdir(args.mesh_folder)
    for ind, folder_path in enumerate(folders_all):
        job_args += [args.mesh_folder + folder_path]
    executor = concurrent.futures.ProcessPoolExecutor(args.num_cpus)
    jobs = [executor.submit(process, job_arg) for job_arg in job_args]
    COUNT = 0
    start_time = time.time()
    for job in concurrent.futures.as_completed(jobs):
        COUNT += 1
        print(
            'Progress: {}/{}, estimated time left: {}'.format(
                COUNT, len(jobs), (time.time() - start_time) / COUNT *
                (len(jobs) - COUNT)
            )
        )
        del job
    executor.shutdown()
