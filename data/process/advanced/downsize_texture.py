# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Downsizing texture images.

Downsize the texture images in the 3D-Front dataset to the specified image size 
(square). Help to speed up simulation.

Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
         Allen Z. Ren (allen.ren@princeton.edu)
"""

import glob
import os
import argparse
from PIL import Image
import concurrent.futures


def process_helper(args):
    return process(args[0], args[1])


def process(full_path, img_size):
    image_files = glob.glob(full_path + '/*.png')
    for image_file in image_files:
        im = Image.open(image_file)
        try:
            im = im.resize((img_size, img_size))
        except:
            continue
        im.save(image_file)
    print('Done: ' + full_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--texture_folder', default='/home/temp/3d-front/3D-FRONT-texture/',
        nargs='?', help='path to the 3D-FUTURE-texture folder'
    )
    parser.add_argument(
        '--img_size', default=128, nargs='?',
        help='image size after downsizing'
    )
    parser.add_argument(
        '--num_cpus',
        default=16,
        nargs='?',
        help='number of cpu threads to use',
    )
    args = parser.parse_args()

    # Run in parallel
    folders_all = os.listdir(args.texture_folder)
    job_args = []
    for ind, folder_path in enumerate(folders_all):
        job_args += [(args.texture_folder + folder_path + '/', args.img_size)]
    executor = concurrent.futures.ProcessPoolExecutor(args.num_cpus)
    jobs = [executor.submit(process_helper, job_arg) for job_arg in job_args]
    COUNT = 0
    for job in concurrent.futures.as_completed(jobs):
        COUNT += 1
        print('Progress: {}/{}'.format(COUNT, len(args)))
        del job
    executor.shutdown()
