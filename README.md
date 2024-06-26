# Image Retrieval

北京交通大学2024春季 邬俊老师课程《计算机视觉基础》

交大视觉印象数据集2024

## 1. Prepare training dataset

Put the data under `${DATA_ROOT}`. The prepared directory would look like:

```bash
${DATA_ROOT}
├── base
│   ├── fh
│   ├── mh
│   ...
│   ├── zx
│   ├── util_pic
├── query
│   ├── fh
│   ├── mh
│   ...
│   ├── zx
```

`${DATA_ROOT}` is set to `./data` by default, which can be modified via hydra command line interface `--data '/your/data/path'`.

Run `split.py` first to split the images in `util_pic` by label (totally 17 classes). Then delete `util_pic`.

Now we have a training dataset (is set to `./data/base` by default) containing **6445** images of **25** categories.

## 2. Fine-tuning

- **Prepare the environment**

```shell
conda activate your_enviroment
cd /your/path/to/your_project
```

- **AlexNet w/o latent layer**

```shell
python finetune.py --model alexnet --batchsize 64 --lr 0.001 --num_epochs 300 --data './your/data/path' --seed 42
```

- **AlexNet w/ latent layer** [CVPRWorkshop2015 Deep Learning of Binary Hash Codes for Fast Image Retrieval](https://homepage.iis.sinica.edu.tw/~kevinlin311.tw/cvprw15.pdf)

```shell
python finetune.py --model alexnet --batchsize 64 --lr 0.001 --num_epochs 300 --data './your/data/path' --seed 42 --latent_layer --latent_size 48
```

- **ResNet-50 w/o latent layer**

```shell
python finetune.py --model resnet --batchsize 64 --lr 0.001 --num_epochs 300 --data './your/data/path' --seed 42
```

- **Memory Usage**

  Running on NVIDIA GeForce RTX 3090.

  <img src="images/image-20240530095653230.png" alt="image-20240530095653230" style="zoom:80%;" />

## 3. Retrieval

Adding `--plot` to the command line will export the retrieved images for each image in the query set. You can find them in `./plots/your_model_name/[20/40/60]`.

- **AlexNet w/o latent layer**

```shell
python retrieval.py --model alexnet --data './your/data/path' --dist [cos/euclidean] [--plot]
```

- **AlexNet w/ latent layer**

```shell
python retrieval.py --model alexnet --data './your/data/path' --latent_layer --dist [cos/euclidean] [--plot]
```

- **ResNet-50 w/o latent layer**

```shell
python retrieval.py --model resnet --data './your/data/path' --dist [cos/euclidean] [--plot]
```

## Example of results

- **AlexNet w/o latent layer (cosine_similarity)**

  `08.png` in `./plots/your_model_name/20`.

  ![08](images/08.png)

  `P@K.png` in `./plots/your_model_name`.

  ![P@K](images/P@K.png)

- **AlexNet w/o latent layer (euclidean_dist)**

  ![P@K](images/P@K-1717172575382-1.png)

- **AlexNet w/ latent layer (binary + cosine_similarity)**

  `08.png` in `./plots/your_model_name/20`.

  ![08](images/08-1717163494741-25.png)

  `P@K.png` in `./plots/your_model_name`.

  ![P@K](images/P@K-1717163049735-22.png)

- **AlexNet w/ latent layer (binary + euclidean_dist)**

  ![P@K](images/P@K-1717172715739-3.png)

- **ResNet-50 w/o latent layer (cosine_similarity)**

  `16.png` in `./plots/your_model_name/60`.

  ![16](images/16.png)

  `P@K.png` in `./plots/your_model_name`.

  ![P@K](images/P@K-1717145786556-7.png)
  
- **ResNet-50 w/o latent layer (euclidean_dist)**

  ![P@K](images/P@K-1717172893845-5.png)
