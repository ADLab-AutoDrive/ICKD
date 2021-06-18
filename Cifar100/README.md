# Cifar100

This repository provides the evaluation code of our paper, the code is based on [RepDistiller](https://github.com/HobbitLong/RepDistiller)(ICLR 2020) codebase to validate our models.

## Benchmark Results on CIFAR-100:

Performance is measured by classification accuracy (%)


| Teacher <br> Student | WRN-40-2 <br> WRN-16-2 | WRN-40-2 <br> WRN-40-1 | ResNet56 <br> ResNet20 | ResNet110 <br> ResNet20 | ResNet110 <br> ResNet32 | ResNet32x4 <br> ResNet8x4 |  Vgg13 <br> Vgg8 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:-----------:|
| Teacher | 75.61 | 75.61 | 72.34 | 74.31 | 74.31 | 79.42 | 74.64 |
| Vanilla | 73.26 | 71.98 | 69.06 | 69.06 | 71.14 | 72.50 | 70.36 |
| KD | 74.92 | 73.54 | 70.66 | 70.67 | 73.08 | 73.33 | 72.98 |
| FitNet | 73.58 | 72.24 | 69.21 | 68.99 | 71.06 | 73.50 | 71.02 |
| AT | 74.08 | 72.77 | 70.55 | 70.22 | 72.31 | 73.44 | 71.43 |
| SP | 73.83 | 72.43 | 69.67 | 70.04 | 72.69 | 72.94 | 72.68 |
| CC | 73.56 | 72.21 | 69.63 | 69.48 | 71.48 | 72.97 | 70.71 |
| VID | 74.11 | 73.30 | 70.38 | 70.16 | 72.61 | 73.09 | 71.23 |
| RKD | 73.35 | 72.22 | 69.61 | 69.25 | 71.82 | 71.90 | 71.48 |
| PKT | 74.54 | 73.45 | 70.34 | 70.25 | 72.61 | 73.64 | 72.88 |
| FSP | 72.91 | 0.00 | 69.95 | 70.11 | 71.89 | 72.62 | 70.23 |
| NST | 73.68 | 72.24 | 69.60 | 69.53 | 71.96 | 73.30 | 71.53 |
| **ICKD-C(w/o KL)** | **75.64** | **74.33** | **71.76** | **71.68** | **73.89** | **75.25** | **73.42** |
| **ICKD-C(Ours)** | **75.57** | **74.63** | **71.69** | **71.91** | **74.11** | **75.48** | **73.88** |

## Runnig

1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. Run distillation by following commands in `scripts/run_cifar_distill.sh`. 
   
    The command for running ICKD (without KL) is something like:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ickd --model_s resnet8x4 -a 0 -b 2.5 --trial 1
    ```
    Combining with KL
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill ickd --model_s resnet8x4 -a 1 -b 2.5 --trial 1
    ```

## Acknowledgement
The work is based on [RepDistiller](https://github.com/HobbitLong/RepDistiller)(ICLR 2020)
