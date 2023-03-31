# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Generate tasks for the Advanced-Dense setting.

Generate random configurations of the rooms by randomly sampling the furniture meshes from the 3D FUTURE dataset. The room dimension is fixed to be 8m square. Also generate the 2D occupancy map of the room.

Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
         Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import argparse
import json
import numpy as np
import trimesh
from shutil import rmtree, copyfile
import random
import time
import matplotlib.pyplot as plt
import concurrent.futures

from utils.process_advanced import add_mat, get_grid_cells_btw, state_lin_to_bin, get_neighbor, slice_mesh, rect_distance
from utils.misc import save_obj


def process_mesh(
    category_all, texture_floor_orig_path, texture_wall_orig_path, task_id,
    args
):
    max_init_goal_attempt = 2000
    max_obs_attempt = 2000
    grid_pitch = 0.05
    free_init_radius = int((0.25+0.75) / grid_pitch)
    free_goal_radius = int((0.25+0.75) / grid_pitch)
    min_init_goal_dist = 5
    min_init_goal_grid = int(min_init_goal_dist / grid_pitch)

    # Room dimensions
    room_height_range = [2.9, 3.9]
    room_height = random.uniform(room_height_range[0], room_height_range[1])

    # raw model name
    if args.use_simplified_mesh:
        raw_model_name = 'raw_model_simplified.obj'
    else:
        raw_model_name = 'raw_model.obj'

    ##############################################

    # Create save folder
    save_path = os.path.join(args.save_task_folder, str(task_id))
    if os.path.isdir(save_path):
        rmtree(save_path)
    os.mkdir(save_path)

    # Generate room mesh
    floor_transform_matrix = [[1, 0, 0, args.room_dim / 2], [0, 1, 0, 0],
                              [0, 0, 1, -0.05], [0, 0, 0, 1]]
    floor = trimesh.creation.box([args.room_dim, args.room_dim, 0.1],
                                 floor_transform_matrix)
    left_wall_transform_matrix = [[1, 0, 0, args.room_dim / 2],
                                  [0, 1, 0, args.room_dim / 2 + 0.05],
                                  [0, 0, 1, room_height / 2], [0, 0, 0, 1]]
    left_wall = trimesh.creation.box([args.room_dim, 0.1, room_height + 0.2],
                                     left_wall_transform_matrix)
    right_wall_transform_matrix = [[1, 0, 0, args.room_dim / 2],
                                   [0, 1, 0, -args.room_dim / 2 - 0.05],
                                   [0, 0, 1, room_height / 2], [0, 0, 0, 1]]
    right_wall = trimesh.creation.box([args.room_dim, 0.1, room_height + 0.2],
                                      right_wall_transform_matrix)
    front_wall_transform_matrix = [[1, 0, 0, args.room_dim + 0.05],
                                   [0, 1, 0, 0], [0, 0, 1, room_height / 2],
                                   [0, 0, 0, 1]]
    front_wall = trimesh.creation.box([
        0.1, args.room_dim + 0.2, room_height + 0.2
    ], front_wall_transform_matrix)
    back_wall_transform_matrix = [[1, 0, 0, -0.05], [0, 1, 0, 0],
                                  [0, 0, 1, room_height / 2], [0, 0, 0, 1]]
    back_wall = trimesh.creation.box([
        0.1, args.room_dim + 0.2, room_height + 0.2
    ], back_wall_transform_matrix)
    room = trimesh.util.concatenate([
        floor, left_wall, right_wall, front_wall, back_wall
    ])
    # room.show()

    # Sample furniture - {'Sofa': 2701, 'Chair': 1775, 'Lighting': 1921, 'Cabinet/Shelf/Desk': 5725, 'Table': 1090, 'Bed': 1124, 'Pier/Stool': 487, 'Others': 1740}
    num_furniture_saved = 0
    category_name_all = ['Sofa', 'Chair', 'Cabinet/Shelf/Desk', 'Table']
    piece_saved_bounds = []
    piece_id_all = []
    piece_pos_all = []
    while num_furniture_saved < args.num_furniture_per_room:
        category_chosen = random.choice(category_name_all)
        num_piece_category_available = len(category_all[category_chosen])
        id_name = category_all[category_chosen][
            random.randint(0, num_piece_category_available - 1)]

        # Some mesh have some issue with vertices...
        try:
            piece = trimesh.load(
                os.path.join(args.mesh_folder, id_name, raw_model_name)
            )
        except:
            continue

        # Make it upright
        piece.apply_transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0],
                               [0, 0, 0, 1]])

        # Check if dimensions too big
        piece_bounds = piece.bounds
        piece_x_dim = piece_bounds[1, 0] - piece_bounds[0, 0]
        piece_y_dim = piece_bounds[1, 1] - piece_bounds[0, 1]
        piece_z_dim = piece_bounds[1, 2] - piece_bounds[0, 2]
        if piece_z_dim > room_height:
            continue

        # Or too small
        if piece_z_dim < 0.5:
            continue
        if piece_x_dim < 0.8 and piece_y_dim < 0.8:
            continue

        # Sample positions
        overlap = True
        obs_attempt = -1
        while obs_attempt < max_obs_attempt and overlap:
            obs_attempt += 1

            x_pos = random.uniform(
                piece_x_dim / 2, args.room_dim - piece_x_dim/2
            )
            y_pos = random.uniform(
                -args.room_dim / 2 + piece_y_dim/2,
                args.room_dim / 2 - piece_y_dim/2
            )
            overlap = False

            # Check gap to other obstacles
            for prev_bounds in piece_saved_bounds:
                a = (
                    piece_bounds[0, 0] + x_pos, piece_bounds[0, 1] + y_pos,
                    piece_bounds[1, 0] + x_pos, piece_bounds[1, 1] + y_pos
                )
                b = (
                    prev_bounds[0, 0], prev_bounds[0, 1], \
                    prev_bounds[1, 0], prev_bounds[1, 1]
                )
                offset = rect_distance(a, b)
                if offset < args.min_obstacle_spacing:
                    overlap = True
                    break

            # Check gap to walls
            # wall_bounds = [
            #     left_wall.bounds, right_wall.bounds, front_wall.bounds,
            #     back_wall.bounds
            # ]
            # for wall_bound in wall_bounds:
            #     a = (
            #         piece_bounds[0, 0] + x_pos, piece_bounds[0, 1] + y_pos,
            #         piece_bounds[1, 0] + x_pos, piece_bounds[1, 1] + y_pos
            #     )
            #     b = (
            #         wall_bound[0, 0], wall_bound[0, 1], \
            #         wall_bound[1, 0], wall_bound[1, 1]
            #     )
            #     offset = rect_distance(a, b)
            #     if offset < min_obs_space:
            #         overlap = True
            #         break

        # Quit
        if obs_attempt == max_obs_attempt:
            return 0

        # desk_height = desk.bounds[1, 1] - desk.bounds[1, 0]
        piece.apply_transform([[1, 0, 0, x_pos], [0, 1, 0, y_pos],
                               [0, 0, 1, 0], [0, 0, 0, 1]])
        piece_bounds = piece.bounds  # update after transform before being saved

        # Add to room
        room = trimesh.util.concatenate([room, piece])
        num_furniture_saved += 1
        piece_id_all += [id_name]
        piece_pos_all += [(x_pos, y_pos, piece_z_dim / 2)]
        piece_saved_bounds += [piece_bounds]
    # room.show()

    # Get 2D occupancy - sometimes cannot voxelize
    try:
        room_mesh = slice_mesh(room)  # only remove floor
        room_voxels = room_mesh.voxelized(pitch=grid_pitch)
    except:
        return 0
    room_voxels_2d = np.max(room_voxels.matrix, axis=2)
    room_voxels_2d[1, :] = 1  # fill in gaps in wall
    room_voxels_2d[-2, :] = 1
    room_voxels_2d[:, 1] = 1
    room_voxels_2d[:, -2] = 1
    # room_voxels.show()
    # plt.show()

    # Sample init and goal
    init_goal_attempt = -1
    while init_goal_attempt < max_init_goal_attempt:
        init_goal_attempt += 1

        N, M = room_voxels_2d.shape
        free_states = np.nonzero((room_voxels_2d == 0).flatten())[0]
        init_state = np.random.choice(free_states)
        goal_state = np.random.choice(free_states)
        init_state_bin = state_lin_to_bin(init_state, [N, M])
        goal_state_bin = state_lin_to_bin(goal_state, [N, M])

        # Check if too close to obstacles
        init_neighbor = get_neighbor(
            room_voxels_2d, init_state_bin, radius=free_init_radius
        )
        goal_neighbor = get_neighbor(
            room_voxels_2d, goal_state_bin, radius=free_goal_radius
        )
        if np.sum(init_neighbor) > 0 or np.sum(goal_neighbor) > 0:
            # print('too close to obstacle')
            continue

        # Check if init and goal too close
        if np.linalg.norm(
            np.array(goal_state_bin) - np.array(init_state_bin)
        ) < min_init_goal_grid:
            # print('init/goal too close')
            continue

        # Check if there is no obstacle between init and goal
        points = get_grid_cells_btw(init_state_bin, goal_state_bin)
        points_voxel = [room_voxels_2d[point[0], point[1]] for point in points]
        if sum(points_voxel) < 5:
            continue
        break

    if init_goal_attempt == max_init_goal_attempt:
        # print('no init/goal found')
        return 0
    init_state = [
        room_voxels.origin[0] + init_state_bin[0] * grid_pitch,
        room_voxels.origin[1] + init_state_bin[1] * grid_pitch,
        random.uniform(0, 2 * np.pi),
    ]
    goal_loc = [
        room_voxels.origin[0] + goal_state_bin[0] * grid_pitch,
        room_voxels.origin[1] + goal_state_bin[1] * grid_pitch,
    ]

    # Sample yaw
    yaw_range = np.pi / 3
    heading_vec = np.array(goal_loc) - np.array(init_state[:2])
    heading = np.arctan2(heading_vec[1], heading_vec[0])
    init_yaw = heading + random.uniform(-yaw_range, yaw_range)
    if init_yaw > np.pi:
        init_yaw -= 2 * np.pi
    elif init_yaw < -np.pi:
        init_yaw += 2 * np.pi
    init_state[2] = init_yaw

    # Add ceiling
    ceiling_transform_matrix = [[1, 0, 0, args.room_dim / 2], [0, 1, 0, 0],
                                [0, 0, 1, room_height + 0.05], [0, 0, 0, 1]]
    ceiling = trimesh.creation.box([args.room_dim, args.room_dim, 0.1],
                                   ceiling_transform_matrix)

    # Export meshes
    wall_path = os.path.join(save_path, 'wall.obj')
    wall = trimesh.util.concatenate([
        ceiling, left_wall, right_wall, front_wall, back_wall
    ])
    wall.export(wall_path)
    floor_path = os.path.join(save_path, 'floor.obj')
    floor.export(floor_path)

    # Set up floor and wall textures
    add_mat(floor_path, floor_path, 'floor_custom')
    add_mat(wall_path, wall_path, 'wall_custom')
    copyfile(args.floor_mtl_path, os.path.join(save_path, 'floor_custom.mtl'))
    copyfile(args.wall_mtl_path, os.path.join(save_path, 'wall_custom.mtl'))
    copyfile(
        texture_floor_orig_path, os.path.join(save_path, 'texture_floor.png')
    )
    copyfile(
        texture_wall_orig_path, os.path.join(save_path, 'texture_wall.png')
    )

    # Debug
    # print('Init: ', init_state)
    # print('Goal: ', goal_loc)
    plt.imshow(room_voxels_2d)
    plt.scatter(init_state_bin[1], init_state_bin[0], s=10, color='red')
    plt.scatter(goal_state_bin[1], goal_state_bin[0], s=10, color='green')
    # for point in points:
    #     plt.scatter(point[1], point[0], s=10, color='blue')
    plt.savefig(os.path.join(save_path, 'task.png'))
    plt.close()

    # Save info
    info = {}
    info['init_state'] = init_state
    info['goal_loc'] = goal_loc
    info['mesh_x_min'] = room_mesh.bounds[0, 0]
    info['mesh_x_max'] = room_mesh.bounds[1, 0]
    info['mesh_y_min'] = room_mesh.bounds[0, 1]
    info['mesh_y_max'] = room_mesh.bounds[1, 1]

    # Save task
    task = {}
    task['task_id'] = task_id
    task['init_state'] = np.array(init_state)
    task['goal_loc'] = np.array(goal_loc)
    task['voxel_grid'] = room_voxels_2d
    task['grid_pitch'] = grid_pitch
    task['mesh_bounds'] = room_mesh.bounds[:, :2].T
    task['piece_id_all'] = piece_id_all
    task['piece_pos_all'] = piece_pos_all
    H, W = room_voxels_2d.shape
    room_map = np.ones((1, H, W), dtype=np.uint8) * 255
    room_map[0] = room_voxels_2d * 255
    task['occupancy_map'] = room_map

    # Pickle task and info
    save_obj([task], os.path.join(save_path, 'task'))
    with open(save_path + '/task.json', 'w') as outfile:
        json.dump(info, outfile)
    return 1


# @reraise_with_stack
def process_mesh_helper(args):
    return process_mesh(args[0], args[1], args[2], args[3], args[4])


if __name__ == "__main__":
    # Process the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_cpus',
        default=16,
        nargs='?',
        help='number of cpu threads to use',
    )
    parser.add_argument(
        '--save_task_folder',
        default='/home/allen/data/pacra/3d-front-tasks-advanced-dense',
        nargs='?', help='path to save the task files'
    )
    parser.add_argument(
        '--mesh_folder', default='/home/temp/3d-front/3D-FUTURE-model',
        nargs='?', help='path to 3D FUTURE dataset'
    )
    parser.add_argument(
        '--texture_folder', default='/home/temp/3d-front/3D-FRONT-texture',
        nargs='?', help='path to texture files'
    )
    parser.add_argument(
        '--floor_mtl_path',
        default='/home/temp/3d-front/default_material/floor_custom.mtl',
        nargs='?', help='path to floor mtl files'
    )
    parser.add_argument(
        '--wall_mtl_path',
        default='/home/temp/3d-front/default_material/wall_custom.mtl',
        nargs='?', help='path to wall mtl files'
    )
    parser.add_argument(
        '--use_simplified_mesh', action='store_true',
        help='use simplified mesh'
    )
    parser.add_argument(
        '--num_room', default=2500, nargs='?',
        help='number of rooms to generate'
    )
    parser.add_argument(
        '--num_furniture_per_room', default=6, nargs='?',
        help='number of furniture per room'
    )
    parser.add_argument(
        '--room_dim', default=8, nargs='?', help='room dimension'
    )
    parser.add_argument(
        '--min_obstacle_spacing', default=0.8, nargs='?',
        help='min obstacle spacing'
    )
    parser.add_argument('--seed', default=42, nargs='?', help='random seed')
    args = parser.parse_args()

    # cfg
    if not os.path.exists(args.save_task_folder):
        os.mkdir(args.save_task_folder)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # super-category: {'Sofa': 2701, 'Chair': 1775, 'Lighting': 1921, 'Cabinet/Shelf/Desk': 5725, 'Table': 1090, 'Bed': 1124, 'Pier/Stool': 487, 'Others': 1740}
    with open(os.path.join(args.mesh_folder, 'model_info.json'), 'r') as f:
        model_info = json.load(f)
    category_all = {}
    for model_ind, model in enumerate(model_info):
        super_category = model['super-category']
        category = model['category']
        id = model['model_id']
        style = model['style']
        theme = model['theme']
        material = model['material']
        if super_category not in category_all:
            category_all[super_category] = [id]
        else:
            category_all[super_category] += [id]

    # Load all available textures
    texture_floor_id_all = []
    texture_wall_id_all = []
    with open(
        os.path.join(args.texture_folder, 'texture_info.json'), 'r',
        encoding='utf-8'
    ) as f:
        data = json.load(f)
        for t in data:
            if t['category'] in [
                'Flooring',
                'Stone',
                'Marble',
            ]:
                texture_floor_id_all += [t['model_id']]
            elif t['category'] in ['Tile', 'Wallpaper', 'Paint']:
                texture_wall_id_all += [t['model_id']]
    print('Number of floor textures available: ', len(texture_floor_id_all))
    print('Number of wall textures available: ', len(texture_wall_id_all))

    # Run parallel in batches until the target number of rooms is reached
    num_batch = 0
    batch_size = 100
    executor = concurrent.futures.ProcessPoolExecutor(args.num_cpus)
    total_room_saved = 0
    while total_room_saved < args.num_room:
        start_time = time.time()
        task_id_batch = [
            num_batch*batch_size + ind for ind in range(batch_size)
        ]
        batch_args = []
        for task_id in task_id_batch:
            # Sample a floor and wall textures for the whole env (rooms share the same texture)
            while 1:
                texture_floor_id = texture_floor_id_all[
                    random.randint(0,
                                   len(texture_floor_id_all) - 1)]
                texture_wall_id = texture_wall_id_all[
                    random.randint(0,
                                   len(texture_wall_id_all) - 1)]
                texture_floor_orig_path = os.path.join(
                    args.texture_folder, texture_floor_id, 'texture.png'
                )
                texture_wall_orig_path = os.path.join(
                    args.texture_folder, texture_wall_id, 'texture.png'
                )
                if os.path.isfile(texture_floor_orig_path
                                 ) and os.path.isfile(texture_wall_orig_path):
                    break  # do not use jpg for now
            batch_args += [(
                category_all, texture_floor_orig_path, texture_wall_orig_path,
                task_id, args
            )]
        print('Processing ', task_id_batch[-1] + 1)
        jobs = [
            executor.submit(process_mesh_helper, job_arg)
            for job_arg in batch_args
        ]
        COUNT = 0
        NUM_ENV_SAVED = 0
        for job in concurrent.futures.as_completed(jobs):
            COUNT += 1
            NUM_ENV_SAVED += job.result()
            print(
                'Progress: {}/{}, estimated hours left (batch): {:.3f}, number of envs saved (batch): {}'
                .format(
                    COUNT, batch_size, (time.time() - start_time) / COUNT *
                    (len(jobs) - COUNT) / 3600, NUM_ENV_SAVED
                )
            )
            del job
        total_room_saved += NUM_ENV_SAVED
        print('Number of processed: ', NUM_ENV_SAVED)
        num_batch += 1
    executor.shutdown()
    print('Total number of envs saved: ', total_room_saved)
