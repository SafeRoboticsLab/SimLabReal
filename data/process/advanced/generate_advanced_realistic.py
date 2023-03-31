# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Generate tasks for the Advanced-Realistic setting.

Post-process house meshes generated from json_to_obj.py. Run Dijkstra to plan possible paths to the goal. Also generate the 2D occupancy map of the room. We have used a few pre-determined parameters in the process_mesh function; feel free to change them to suit your needs.

Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
         Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import argparse
import random
import glob
import json
import time
import numpy as np
import concurrent.futures
from shutil import rmtree
import matplotlib.pyplot as plt
from PIL import Image
import trimesh
from shapely.geometry import Point

from data.process.advanced.dijkstra import Dijkstra
from utils.misc import save_obj
from utils.process_advanced import get_grid_cells_btw, state_lin_to_bin, state_bin_to_lin, apply_move, check_free, get_neighbor, slice_mesh, moves


def build_TR(grid):
    N, M = grid.shape
    num_state = N * M
    num_action = 4

    # maximum likely versions
    Tml = np.zeros([num_state, num_action], 'i')  # Tml[s, a] --> next state

    for i in range(N):
        for j in range(M):
            state_coord = np.array([i, j])
            state = state_bin_to_lin(state_coord, [N, M])

            # build T and R
            for act in range(num_action):
                neighbor_coord = apply_move(state_coord, moves[act])
                if not check_free(grid, neighbor_coord):
                    neighbor_coord[:2] = [
                        i, j
                    ]  # dont move if obstacle or edge of world

                neighbor = state_bin_to_lin(neighbor_coord, [N, M])
                Tml[state, act] = neighbor
    return Tml


def process_mesh(house_path, args):
    s1 = time.time()
    max_time = 600  # 10 minutes
    num_max_attempt_ratio = 1000
    num_max_travese_attempt = 1000000
    grid_pitch = 0.1
    min_init_goal_dist = 3
    max_init_goal_dist = 7
    min_init_goal_grid = int(min_init_goal_dist / grid_pitch)
    max_init_goal_grid = int(max_init_goal_dist / grid_pitch)
    cost_factor = 10  # use large factor to make sure Dijstra finds path evenly between obstacles
    free_init_radius = 8
    free_goal_radius = 4
    cost_passage_radius = 8
    free_passage_radius = 4
    room_area_ratio = 50
    yaw_range = np.pi / 6
    max_room_size = 250
    room_free_init_dim = 0.50
    max_num_room = 1
    num_max_target_env = 2
    normalized_map_dim = 320
    save_parent_path = os.path.join(
        args.save_task_folder,
        house_path.split('/')[-1]
    )
    if os.path.isdir(save_parent_path):
        rmtree(save_parent_path)
    os.mkdir(save_parent_path)

    # Load all rooms
    room_mesh_all = []
    room_bound_all = []
    room_name_all = []
    room_floor_polygon_all = []
    for room_ind, room_name in enumerate(os.listdir(house_path)):
        room_obj_all = glob.glob(os.path.join(house_path, room_name, '*.obj'))
        mesh_all = []
        room_floor_polygon_all += [
            None
        ]  # in case no floor mesh for the room - not sure if ever happens
        for obj_path in room_obj_all:
            try:  # error about vertices index out of bound
                mesh = trimesh.load(obj_path)
            except:
                continue

            # Rotate s.t. z up
            align_matrix = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0],
                                     [0, 0, 0, 1]])
            mesh.apply_transform(align_matrix)
            mesh_dim = mesh.bounds

            # Store floor polygons - error about too many disconnected groups
            try:
                if 'floor' in obj_path:
                    polygon = trimesh.path.polygons.projected(
                        mesh, [0, 0, 1], origin=None, apad=1e-05, tol_dot=0.01,
                        max_regions=200
                    )
                    room_floor_polygon_all[-1] = polygon
            except:
                continue

            # Ignore meshes with trivial height - including floors
            if mesh_dim[1, 2] - mesh_dim[0, 2] < 1e-2:
                continue
            mesh_all += [mesh]
        if len(mesh_all) > 0:
            room_mesh = trimesh.util.concatenate(mesh_all)
            room_bounds = room_mesh.bounds[:, :2]
            room_mesh_all += [room_mesh]
            room_bound_all += [room_bounds]
            room_name_all += [room_name]

            # Quit if a single room is too big
            room_area = room_bounds[:, 0].ptp() * room_bounds[:, 1].ptp()
            if room_area > max_room_size:
                # print('Room too big')
                return (0, time.time() - s1)
    house_mesh = trimesh.util.concatenate(room_mesh_all)

    # Get 2D occupancy grid  sometimes cannot voxelize
    try:
        house_mesh_below = slice_mesh(house_mesh)
        house_mesh_below_bounds = house_mesh_below.bounds[:, :2]  # x,y only
        voxels = house_mesh_below.voxelized(pitch=grid_pitch)
        voxel_2d = np.max(voxels.matrix, axis=2)
    except:
        # print('cannot voxelize')
        return (0, time.time() - s1)

    # Get house area
    house_area = np.prod(
        house_mesh_below_bounds[1] - house_mesh_below_bounds[0]
    )
    num_target_env = min(
        max(int(house_area / room_area_ratio), 1), num_max_target_env
    )
    num_max_attempt = num_target_env * num_max_attempt_ratio
    num_attempt = 0
    num_save_env = 0

    # Construct graph for computing paths
    N, M = voxel_2d.shape
    num_state = N * M
    num_action = 4
    Tml = build_TR(voxel_2d)  # transition model
    G = {i: {} for i in range(num_state)}
    for a in range(num_action):
        for s in range(num_state):
            snext = Tml[s, a]

            # Get cost based on adjacent occupancy
            snext_bin = state_lin_to_bin(snext, [N, M])
            neighbor = get_neighbor(
                voxel_2d, snext_bin, radius=cost_passage_radius
            )
            snext_cost = np.sum(neighbor) / ((2*cost_passage_radius + 1)**
                                             2) * cost_factor + 1

            # Assign cost to next state
            if s != snext:
                G[snext][s] = snext_cost  # before cost always equals 1
    graph = G

    # Sample init/goal states
    saved_room_ind_comb_all = [
    ]  # if an new env found has the same set of rooms as one env before, do not save the new env since we should count it as the same env as before
    s3 = time.time()
    while time.time() - s3 < max_time and num_attempt < num_max_attempt:

        # Count attempt
        env_found = False
        num_attempt += 1
        task = {}
        task['mesh_path'] = house_path

        for _ in range(num_max_travese_attempt):
            free_states = np.nonzero((voxel_2d == False).flatten())[0]
            init_state = np.random.choice(free_states)
            goal_state = np.random.choice(free_states)
            init_state_bin = state_lin_to_bin(init_state, [N, M])
            goal_state_bin = state_lin_to_bin(goal_state, [N, M])

            # Check if init or goal too close to obstacles
            init_neighbor = get_neighbor(
                voxel_2d, init_state_bin, radius=free_init_radius
            )
            goal_neighbor = get_neighbor(
                voxel_2d, goal_state_bin, radius=free_goal_radius
            )
            if np.sum(init_neighbor) > free_init_radius or np.sum(
                goal_neighbor
            ) > free_goal_radius:
                continue

            # Check if init and goal too close or too far
            init_goal_grid = np.linalg.norm(
                np.array(goal_state_bin) - np.array(init_state_bin)
            )
            if init_goal_grid < min_init_goal_grid or init_goal_grid > max_init_goal_grid:
                # print('init/goal too close')
                continue

            # Check if there is obstacle between init and goal
            points = get_grid_cells_btw(init_state_bin, goal_state_bin)
            points_voxel = [voxel_2d[point[0], point[1]] for point in points]
            if sum(points_voxel) < 1:
                # print('no obstacle')
                continue

            # check if path exists from start to goal, if not, pick a new set
            D, P = Dijkstra(
                graph, goal_state
            )  # map of distances and predecessor
            if init_state in D:
                cost = D[init_state]
                env_found = True
                break
        if not env_found:
            # print('No env found!')
            continue

        # Get trajectory
        traj = []
        init_state_copy = init_state
        goal_state_copy = goal_state
        while 1:
            traj.append(init_state_copy)
            if init_state_copy == goal_state_copy:
                break
            init_state_copy = P[init_state_copy]

        # Get traversed rooms - make sure state in rooms (excluding other rooms) and state not too close to obstacle on both sides
        traversed_room_names = []
        traversed_room_ind = []
        valid_traj = True
        for state_ind, state in enumerate(traj):
            state_bin = state_lin_to_bin(state, [N, M])
            state = [
                voxels.origin[0] + state_bin[0] * grid_pitch,
                voxels.origin[1] + state_bin[1] * grid_pitch,
            ]

            # Check neighbors
            neighbor = get_neighbor(
                voxel_2d, state_bin, radius=free_passage_radius
            )
            if np.sum(neighbor) > free_passage_radius:
                valid_traj = False
                break

            state_in_room = False  # check if state in any of the rooms - prevent traversing into empty space between rooms
            for room_ind, room_bound in enumerate(room_bound_all):
                if state[0] > room_bound[0, 0] and \
                   state[0] < room_bound[1, 0] and \
                   state[1] > room_bound[0, 1] and \
                   state[1] < room_bound[1, 1] and 'Other' not in room_name_all[room_ind]:  # do not use other room - often empty space around the house
                    room_name = room_name_all[room_ind]
                    state_in_room = True
                    if room_name not in traversed_room_names:
                        traversed_room_names += [room_name]
                        traversed_room_ind += [room_ind]

                        # Check if init_state is near boundary of the room
                        if state_ind == 0:
                            if abs(state[0] - room_bound[0, 0]) < room_free_init_dim or \
                                abs(state[0] - room_bound[1, 0]) < room_free_init_dim or \
                                abs(state[1] - room_bound[0, 1]) < room_free_init_dim or \
                                abs(state[1] - room_bound[1, 1]) < room_free_init_dim:
                                valid_traj = False
                                # print('too close to room boundary')
                        break

            # Do not save this env
            if not state_in_room:
                valid_traj = False
            if not valid_traj:
                break
        if not valid_traj:
            # print('State not in any room!')
            continue

        # Limit number of rooms
        if len(traversed_room_ind) > max_num_room:
            continue

        # Check if this room setup is sampled before
        traversed_room_ind.sort()
        if traversed_room_ind in saved_room_ind_comb_all:
            continue

        # Check if only balcony
        if len(traversed_room_names) == 1 and 'alcony' in traversed_room_names:
            continue

        # Check if sampled init/goal in floor polygon
        init_state = [
            voxels.origin[0] + init_state_bin[0] * grid_pitch,
            voxels.origin[1] + init_state_bin[1] * grid_pitch,
            0,
        ]
        goal_loc = [
            voxels.origin[0] + goal_state_bin[0] * grid_pitch,
            voxels.origin[1] + goal_state_bin[1] * grid_pitch,
        ]
        traversed_room_polygon_all = [
            room_floor_polygon_all[ind] for ind in traversed_room_ind
        ]
        init_on_floor = False
        goal_on_floor = False
        for polygon in traversed_room_polygon_all:
            if polygon is None:  # no floor mesh?
                continue
            if polygon.contains(Point(init_state[:2])):
                init_on_floor = True
            if polygon.contains(Point(goal_loc)):
                goal_on_floor = True
        if not init_on_floor or not goal_on_floor:
            # print('not on floor!')
            continue

        # Save this env from this point on
        saved_room_ind_comb_all += [traversed_room_ind]
        task['room_names'] = traversed_room_names
        env_save_path = os.path.join(save_parent_path, str(num_save_env))
        if os.path.isdir(env_save_path):
            rmtree(env_save_path)
        os.mkdir(env_save_path)

        # Plot traj
        plt.imshow(voxel_2d)
        plt.scatter(init_state_bin[1], init_state_bin[0], s=50, color='red')
        plt.scatter(goal_state_bin[1], goal_state_bin[0], s=50, color='green')
        for state in traj:
            state_bin = state_lin_to_bin(state, [N, M])
            plt.scatter(state_bin[1], state_bin[0], s=10, color='blue')
        plt.savefig(os.path.join(env_save_path, 'task.png'))
        plt.close()

        # Sample yaw relatively facing the goal
        heading_vec = np.array(goal_loc) - np.array(init_state[:2])
        heading = np.arctan2(heading_vec[1], heading_vec[0])
        init_yaw = heading + random.uniform(-yaw_range, yaw_range)
        if init_yaw > np.pi:
            init_yaw -= 2 * np.pi
        elif init_yaw < -np.pi:
            init_yaw += 2 * np.pi
        init_state[2] = init_yaw
        task['init_state'] = np.array(init_state)
        task['goal_loc'] = np.array(goal_loc)

        # Get occupancy grid of truncated mesh
        truncated_mesh = trimesh.util.concatenate([
            room_mesh_all[ind] for ind in traversed_room_ind
        ])
        truncated_mesh_below = slice_mesh(truncated_mesh)
        truncated_mesh_below_bounds = truncated_mesh_below.bounds  # not necessarily same as truncated_mesh_bound
        truncated_voxels = truncated_mesh_below.voxelized(pitch=grid_pitch)
        truncated_voxel_2d = np.max(truncated_voxels.matrix, axis=2)

        # Save grid and pitch to task - for visualization
        task['voxel_grid'] = truncated_voxel_2d
        task['grid_pitch'] = grid_pitch
        task['mesh_bounds'] = truncated_mesh_below_bounds[:, :2].T

        # Save info
        info = {}
        info['traversed_room_names'] = traversed_room_names
        info['cost'] = cost
        info['init_state'] = init_state
        info['goal_loc'] = goal_loc
        info['mesh_x_min'] = truncated_mesh_below_bounds[0, 0]
        info['mesh_x_max'] = truncated_mesh_below_bounds[1, 0]
        info['mesh_y_min'] = truncated_mesh_below_bounds[0, 1]
        info['mesh_y_max'] = truncated_mesh_below_bounds[1, 1]

        # Save normalized map and state conversion
        normalized_map_img = Image.fromarray(
            np.uint8(truncated_voxel_2d * 255)
        ).resize((normalized_map_dim, normalized_map_dim), Image.LANCZOS)
        normalized_map_img_arr = np.array(
            normalized_map_img
        )  # C=1, HxW, 0-255
        task['normalized_map'] = normalized_map_img_arr
        task['map_dimension'] = normalized_map_dim

        # Save task and info
        save_obj([task], os.path.join(env_save_path, 'task'))
        with open(env_save_path + '/task.json', 'w') as outfile:
            json.dump(info, outfile)
        num_save_env += 1

        # Stop if enough envs sampled
        if num_save_env == num_target_env:
            break
    return (num_save_env, time.time() - s1)


def process_mesh_helper(args):
    return process_mesh(args[0], args[1])


if __name__ == "__main__":
    # Process the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_cpus',
        default=20,
        nargs='?',
        help='number of cpu threads to use',
    )
    parser.add_argument(
        '--save_task_folder',
        default='/home/temp/3d-front/3d-front-tasks-advanced-realistic',
        nargs='?', help='path to save the task files'
    )
    parser.add_argument(
        '--house_folder', default='/home/temp/3d-front/3d-front-house-meshes',
        nargs='?', help='path to house OBJ meshes'
    )
    args = parser.parse_args()
    if not os.path.exists(args.save_task_folder):
        os.mkdir(args.save_task_folder)

    # Process in batches
    house_path_all = os.listdir(args.house_folder)
    all_args = [(os.path.join(args.house_folder, house_name), args)
                for house_name in house_path_all]
    num_mesh = len(all_args)
    batch_size = 6813
    num_batch = int(np.ceil(num_mesh / batch_size))
    print('Number of batches: ', num_batch)
    executor = concurrent.futures.ProcessPoolExecutor(args.num_cpus)
    total_envs_saved = 0
    for batch_ind in range(0, num_batch):
        start_time = time.time()
        print('Batch: ', batch_ind)
        batch_args = all_args[batch_ind * batch_size:(batch_ind+1)
                              * batch_size]
        jobs = [
            executor.submit(process_mesh_helper, job_arg)
            for job_arg in batch_args
        ]
        COUNT = 0
        NUM_ENV_SAVED = 0
        for job in concurrent.futures.as_completed(jobs):
            COUNT += 1
            NUM_ENV_SAVED += job.result()[0]
            print(
                'Progress: {}/{}, estimated hours left (batch): {:.3f}, number of envs saved (batch): {}'
                .format(
                    COUNT, batch_size, (time.time() - start_time) / COUNT *
                    (len(jobs) - COUNT) / 3600, NUM_ENV_SAVED
                )
            )
            del job
        total_envs_saved += NUM_ENV_SAVED
    executor.shutdown()
    print('Total number of envs saved: ', total_envs_saved)
