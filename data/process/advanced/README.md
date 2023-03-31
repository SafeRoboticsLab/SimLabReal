## Generate dataset for the Advanced environments

1. Download the 3D Front and 3D Future dataset upon request [here](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future).

2. (Optional) Simplify the OBJ meshes in the 3D Future dataset (16565 pieces of furniture) using Blender script from [Bakri](https://github.com/HusseinBakri) - this can take a while:

    ```console
    sudo apt-get install blender
    python data/process/advanced/simplify_mesh.py --mesh_folder=[path_to_3D_FUTURE_MODEL] --num_cpus=.
    ```

3. (Optional) Downsize the texture image files in the 3D-Front dataset:

    ```console
    python data/process/advanced/downsize_texture.py --texture_folder=[path_to_3D_FRONT_TEXTURE] --img_size=128 --num_cpus=.
    ```

4. (For the Advanced-Dense setting), generate >2000 room meshes by randomly sampling furniture meshes from the 3D Future dataset, and save the task configurations in a new folder - this can take a while:

    ```console
    python data/process/advanced/generate_advanced_dense.py --num_cpus=. --save_task_folder=[path_to_dense_task] --mesh_folder=[path_to_3D_FUTURE_MODEL] --texture_folder=[path_to_3D_FRONT_TEXTURE] --floor_mtl_path=. --wall_mtl_path=. --use_simplified_mesh=. --num_room=2500
    ```

5. (For the Advanced-Realistic setting), generate 6813 house meshes by reading the json files in the 3D Front dataset and merging the meshes from the 3D Future dataset, and save them in a new folder: 
    
    ```console
    python data/process/advanced/json_to_obj.py --num_cpus=. --save_folder=[house_folder] --future_folder=[path_to_3D_FUTURE_MODEL] --json_folder=[path_to_3D_FRONT] --texture_folder=[path_to_3D_FRONT_TEXTURE] --floor_mtl_path=. --wall_mtl_path=. --use_simplified_mesh=.
    ```

6. (For the Advanced-Realistic setting), post-process the house meshes, and save the task configurations in a new folder - this can take a while and requires a siginicant amount of RAM. Roughly 2500 configurations would be saves.

    ```console
    python data/process/advanced/generate_advanced_realistic.py --num_cpus=. --save_task_folder=[path_to_realistic_task] --house_folder=[house_folder]
    ```

7. (For the Advanced-Dense setting), split the generated task configurations to train and test datasets.

    ```console
    python data/process/advanced/split_advanced_dense.py --save_folder=[path_to_dense_dataset] --task_folder=[path_to_dense_task]
    ```

8. (For the Advanced-Realistic setting), split the generated task configurations to train and test datasets.

    ```console
    python data/process/advanced/split_advanced_dense.py --save_folder=[path_to_realistic_dataset] --task_folder=[path_to_realistic_task]
    ```