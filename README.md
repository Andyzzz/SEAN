# Modified SEAN: get test results in batches instead of UI

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.2.0](https://img.shields.io/badge/pytorch-1.2.0-green.svg?style=plastic)
![pyqt5 5.13.0](https://img.shields.io/badge/pyqt5-5.13.0-green.svg?style=plastic)

![image](./docs/assets/Teaser.png)
**Figure:** *Face image editing controlled via style images and segmentation masks with SEAN*



> **SEAN: Image Synthesis with Semantic Region-Adaptive Normalization** <br>
> Peihao Zhu, Rameen Abdal, Yipeng Qin, Peter Wonka <br>
> *Computer Vision and Pattern Recognition **CVPR 2020, Oral***



## Installation

Clone this repo.
```bash
git clone https://github.com/ZPdesu/SEAN.git
cd SEAN/
```

This code requires PyTorch, python 3+ and Pyqt5. Please install dependencies by
```bash
pip install -r requirements.txt
```

This model requires a lot of memory and time to train. To speed up the training, we recommend using 4 V100 GPUs


## Dataset Preparation

This code uses [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) and [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset. The prepared dataset can be directly downloaded [here](https://drive.google.com/file/d/1TKhN9kDvJEcpbIarwsd1_fsTR2vGx6LC/view?usp=sharing). After unzipping, put the entire CelebA-HQ folder in the datasets folder. The complete directory should look like `./datasets/CelebA-HQ/train/` and `./datasets/CelebA-HQ/test/`.


## Generating Images Using Pretrained Models

Once the dataset is prepared, the reconstruction results be got using pretrained models.


1. Create `./checkpoints/` in the main folder and download the tar of the pretrained models from the [Google Drive Folder](https://drive.google.com/file/d/1UMgKGdVqlulfgOBV4Z0ajEwPdgt3_EDK/view?usp=sharing). Save the tar in `./checkpoints/`, then run

    ```
    cd checkpoints
    tar CelebA-HQ_pretrained.tar.gz
    cd ../
    ```

2. Generate the reconstruction results using the pretrained model.
	```bash
   python test.py --name CelebA-HQ_pretrained --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --gpu_ids 0
    ```

3. The reconstruction images are saved at `./results/CelebA-HQ_pretrained/` and the corresponding style codes are stored at `./styles_test/style_codes/`.

4. Pre-calculate the mean style codes for the UI mode. The mean style codes can be found at `./styles_test/mean_style_code/`.

	```bash
    python calculate_mean_style_code.py
    ```


## Training New Models

To train the new model, you need to specify the option `--dataset_mode custom`, along with `--label_dir [path_to_labels] --image_dir [path_to_images]`. You also need to specify options such as `--label_nc` for the number of label classes in the dataset, and `--no_instance` to denote the dataset doesn't have instance maps.


```bash
python train.py --name [experiment_name] --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/train/labels --image_dir datasets/CelebA-HQ/train/images --label_nc 19 --no_instance --batchSize 32 --gpu_ids 0,1,2,3
```

If you only have single GPU with small memory, please use `--batchSize 2 --gpu_ids 0`.


## Batch Test Introduction

We provide a convenient code for the users to test images in batches.You need to:

1. Create `./checkpoints/` in the main folder and download the tar of the pretrained models from the [Google Drive Folder](https://drive.google.com/file/d/1UMgKGdVqlulfgOBV4Z0ajEwPdgt3_EDK/view?usp=sharing). Save the tar in `./checkpoints/`, then run

    ```
    cd checkpoints
    tar CelebA-HQ_pretrained.tar.gz
    cd ../
    ```
2. Set the test options in `options/test_options.py`, put the test images into `datasets/xxx/test_img` and semantic segmentation masks into `datasets/xxx/test_label`.

3. Extract the style codes from test images and generate the reconstruction results using the pretrained model.
	```bash
   python test.py --name CelebA-HQ_pretrained --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test_label --image_dir datasets/CelebA-HQ/test_img --label_nc 19 --no_instance --gpu_ids 0
    ```

4. Pre-calculate the mean style codes for the UI mode. The mean style codes can be found at `./styles_test/mean_style_code/`.

	```bash
    python calculate_mean_style_code.py
    ```

5. Classify the style codes via semantic labels and save in `styles_random3`
```bash
    python style_random.py
    ```
6. Generate the results with random hair styles.
```bash
    python test_batch.py
    ```



