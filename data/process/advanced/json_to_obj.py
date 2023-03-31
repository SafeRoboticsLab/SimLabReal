# --------------------------------------------------------
# Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees
# https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, allen.ren@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Generate OBJ meshes from 3D-Front dataset.

Convert JSON configuration file of the houses from the 3D-Front dataset to OBJ file of the houses. Modified from https://github.com/3D-FRONT-FUTURE/3D-FRONT-ToolBox.

Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
         Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import argparse
import json
import trimesh
import numpy as np
import math
import igl
from shutil import copyfile, rmtree
import random
from pathos.multiprocessing import ProcessPool


use_mat = "mtllib {}.mtl\nusemtl default\n"


def add_mat(input_mesh, output_mesh, mat_name):
    with open(input_mesh, "r") as fin:
        lines = fin.readlines()
    for l in lines:
        if l == "mtllib {}.mtl\n".format(mat_name):
            return
    with open(output_mesh, "w") as fout:
        fout.write(use_mat.format(mat_name))
        for line in lines:
            if not line.startswith("o") and not line.startswith("s"):
                fout.write(line)


def split_path(paths):
    filepath, tempfilename = os.path.split(paths)
    filename, extension = os.path.splitext(tempfilename)
    return filepath, filename, extension


def write_obj_with_tex(savepath, vert, face, vtex, ftcoor, imgpath=None):
    filepath2, filename, extension = split_path(savepath)
    with open(savepath, 'w') as fid:
        fid.write('mtllib ' + filename + '.mtl\n')
        fid.write('usemtl a\n')
        for v in vert:
            fid.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for vt in vtex:
            fid.write('vt %f %f\n' % (vt[0], vt[1]))
        face = face + 1
        ftcoor = ftcoor + 1
        for f, ft in zip(face, ftcoor):
            fid.write(
                'f %d/%d %d/%d %d/%d\n' %
                (f[0], ft[0], f[1], ft[1], f[2], ft[2])
            )
    filepath, filename2, extension = split_path(imgpath)
    if os.path.exists(
        imgpath
    ) and not os.path.exists(filepath2 + '/' + filename + extension):
        copyfile(imgpath, filepath2 + '/' + filename + extension)
    if imgpath is not None:
        with open(filepath2 + '/' + filename + '.mtl', 'w') as fid:
            fid.write('newmtl a\n')
            fid.write('map_Kd ' + filename + extension)


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc+ad), 2 * (bd-ac)],
                     [2 * (bc-ad), aa + cc - bb - dd, 2 * (cd+ab)],
                     [2 * (bd+ac), 2 * (cd-ab), aa + dd - bb - cc]])


def process_json_helper(args):
    process_json(args[0], args[1], args[2], args[3])


def process_json(
    json_path, texture_floor_orig_path, texture_wall_orig_path, args
):
    # In case not using the meshes at the end
    rm_mesh = False

    # raw model name
    if args.use_simplified_mesh:
        raw_model_name = 'raw_model_simplified.obj'
    else:
        raw_model_name = 'raw_model.obj'

    # Load data
    with open(
        os.path.join(args.json_folder, json_path), 'r', encoding='utf-8'
    ) as f:
        data = json.load(f)
        model_jid = []
        model_uid = []
        model_bbox = []
        mesh_uid = []
        mesh_xyz = []
        mesh_faces = []
        mesh_types = []

        # House
        house_id = json_path[:-5]
        house_path = os.path.join(args.save_folder, house_id)
        if not os.path.exists(house_path):
            os.mkdir(house_path)
        for ff in data['furniture']:
            if 'valid' in ff and ff['valid']:
                model_uid.append(ff['uid'])
                model_jid.append(ff['jid'])
                model_bbox.append(ff['bbox'])
        for mm in data['mesh']:
            mesh_uid.append(mm['uid'])
            mesh_xyz.append(np.reshape(mm['xyz'], [-1, 3]))
            mesh_faces.append(np.reshape(mm['faces'], [-1, 3]))
            mesh_types.append(mm['type'])
        scene = data['scene']
        room = scene['room']
        for r in room:
            room_id = r['instanceid']
            room_path = house_path + '/' + room_id
            # print('Room: ', room_id)

            meshes_wall = []
            meshes_floor = []
            meshes = []
            if not os.path.exists(room_path):
                os.mkdir(room_path)
            children = r['children']
            number = 1
            for c in children:
                ref = c['ref']
                type = 'f'
                try:
                    idx = model_uid.index(ref)
                    if os.path.exists(
                        os.path.join(args.future_folder, model_jid[idx])
                    ):
                        v, vt, _, faces, ftc, _ = igl.read_obj(
                            os.path.join(
                                args.future_folder, model_jid[idx],
                                raw_model_name
                            )
                        )
                except:
                    try:
                        idx = mesh_uid.index(ref)
                    except:
                        continue
                    v = mesh_xyz[idx]
                    faces = mesh_faces[idx]
                    type = 'm'
                    mtype = mesh_types[idx]
                pos = c['pos']
                rot = c['rot']
                scale = c['scale']
                v = v.astype(np.float64) * scale
                ref = [0, 0, 1]
                axis = np.cross(ref, rot[1:])
                theta = np.arccos(np.dot(ref, rot[1:])) * 2
                if np.sum(axis) != 0 and not math.isnan(theta):
                    R = rotation_matrix(axis, theta)
                    v = np.transpose(v)
                    v = np.matmul(R, v)
                    v = np.transpose(v)
                v = v + pos
                if type == 'f':
                    write_obj_with_tex(
                        os.path.join(
                            room_path,
                            str(number) + '_' + model_jid[idx] + '.obj'
                        ), v, faces, vt, ftc,
                        os.path.join(
                            args.future_folder, model_jid[idx], 'texture.png'
                        )
                    )
                    number = number + 1
                else:
                    mesh = trimesh.Trimesh(v, faces)
                    if 'Floor' in mtype:
                        meshes_floor.append(mesh)
                    elif 'Wall' in mtype:
                        meshes_wall.append(mesh)
                    elif 'lab' not in mtype:  # slab
                        meshes.append(mesh)

            if len(meshes) > 0:
                temp = trimesh.util.concatenate(meshes)
                mesh_path = os.path.join(room_path, 'mesh.obj')
                temp.export(mesh_path)

            if len(meshes_floor) > 0:
                temp = trimesh.util.concatenate(meshes_floor)

                mesh_path = os.path.join(room_path, 'mesh_floor.obj')
                temp.export(mesh_path)
                add_mat(mesh_path, mesh_path, 'floor_custom')
                copyfile(
                    args.floor_mtl_path,
                    os.path.join(
                        args.save_folder, json_path[:-5], room_id,
                        'floor_custom.mtl'
                    )
                )
                copyfile(
                    texture_floor_orig_path,
                    os.path.join(room_path, 'texture_floor.png')
                )

            if len(meshes_wall) > 0:
                temp = trimesh.util.concatenate(meshes_wall)
                mesh_path = os.path.join(room_path, 'mesh_wall.obj')
                temp.export(mesh_path)
                add_mat(mesh_path, mesh_path, 'wall_custom')
                copyfile(
                    args.wall_mtl_path,
                    os.path.join(room_path, 'wall_custom.mtl')
                )
                copyfile(
                    texture_wall_orig_path,
                    os.path.join(room_path, 'texture_wall.png')
                )
        if rm_mesh:
            rmtree(house_path)


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
        '--save_folder', default='/home/temp/3d-front/3d-front-house-meshes',
        nargs='?', help='path to save house OBJ meshes'
    )
    parser.add_argument(
        '--mesh_folder', default='/home/temp/3d-front/3D-FUTURE-model',
        nargs='?', help='path to 3D FUTURE dataset'
    )
    parser.add_argument(
        '--json_folder', default='/home/temp/3d-front/3D-FRONT', nargs='?',
        help='path to 3D FRONT dataset (in json format))'
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
    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # Get all floor and wall textures
    texture_floor_id_all = []
    texture_wall_id_all = []
    with open(
        os.path.join(args.texture_folder, 'texture_info.json'), 'r',
        encoding='utf-8'
    ) as f:
        data = json.load(f)
        for t in data:
            if t['category'] in [
                'Flooring', 'Stone', 'Wood', 'Marble', 'Solid wood flooring'
            ]:
                texture_floor_id_all += [t['model_id']]
            elif t['category'] in ['Tile', 'Wallpaper', 'Paint']:
                texture_wall_id_all += [t['model_id']]
    print('Number of floor textures available: ', len(texture_floor_id_all))
    print('Number of wall textures available: ', len(texture_wall_id_all))

    # Go through all room json configs
    json_paths = os.listdir(args.json_folder)

    # run parallel for all json paths
    job_args = []
    for json_path in json_paths:
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

        job_args += [
            (json_path, texture_floor_orig_path, texture_wall_orig_path, args)
        ]
    pool = ProcessPool(nodes=args.num_cpus)
    pool.map(process_json_helper, job_args)
