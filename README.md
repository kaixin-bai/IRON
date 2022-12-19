# IRON: Inverse Rendering by Optimizing Neural SDFs and Materials from Photometric Images

Note: this repo is still under construction.

Project page: <https://kai-46.github.io/IRON-website/>

![example results](./readme_resources/inputs_outputs.png)

## Usage

### Create environment

```shell
git clone https://github.com/Kai-46/iron.git && cd iron && . ./create_env.sh
```

### Download data

```shell
. ./download_data.sh
mkdir data_flashlight
# 更改数据集文件夹名字为Luan_et_al_2021
cp -r Luan_et_al_2021 ./data_flashlight
```

### Training and testing

```shell
# . ./train_scene.sh drv/dragon
# 删除train_scene.sh里面所有的data_flashlight
python3 render_volume.py --mode train --conf ./confs/womask_iron.conf --case  Luan_et_al_2021/xmen
```

Once training is done, you will see the recovered mesh and materials under the folder ```./exp_iron_stage2/drv/dragon/mesh_and_materials_50000/```. At the same time, the rendered test images are under the folder ``````./exp_iron_stage2/drv/dragon/render_test_50000/``````

### Relight the 3D assets using envmaps

Check ```test_mitsuba/render_rgb_envmap_mat.py```.

### Evaluation

Check ```evaluation/eval_mesh.py``` and ```evaluation/eval_image_folder.py```.

### Render synthetic data using Mitsuba

Check ```render_synthetic_data/render_rgb_flash_mat.py```. To make renderings more shiny, try scaling up the specular albedo and scaling down the specular roughness; to make renderings more diffuse, try the opposite.

### Camera parameters convention

We use the OpenCV camera convention just like [NeRF++](https://github.com/Kai-46/nerfplusplus); you might want to use the camera visualization and debugging tools in that codebase to inspect if there's any issue with the camera parameters. Note we also assume the objects are inside the unit sphere.

## Citations

```
@inproceedings{iron-2022,
  title={IRON: Inverse Rendering by Optimizing Neural SDFs and Materials from Photometric Images},
  author={Zhang, Kai and Luan, Fujun and Li, Zhengqi and Snavely, Noah},
  booktitle={IEEE Conf. Comput. Vis. Pattern Recog.},
  year={2022}
}
```

## Example results

<https://user-images.githubusercontent.com/21653654/174612222-91302de2-34f2-429c-b53e-f78f140873c4.mp4>

![example results](./readme_resources/assets_lowres.png)

## Acknowledgements

We would like to thank the authors of [IDR](https://github.com/lioryariv/idr) and [NeuS](https://github.com/Totoro97/NeuS) for open-sourcing their projects.
