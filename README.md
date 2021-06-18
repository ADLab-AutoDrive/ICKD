# ICKD
This repository provides code of our paper "Exploring Inter-Channel Correlation for Diversity-preserved Knowledge Distillation". We provide training code on Cifar100 and evaluation code on ImageNet & Pascal VOC. The remaining training code will be released after the paper is accepted.

## Preparation
1. Download all checkpoints from https://drive.google.com/drive/folders/1ZvwEAVJurTXSuPL_0HylHMNUGbG1zl3U?usp=sharing
2. Build a docker image using the provided Dockerfile. All code should be run in the docker image.

## Running
Go into directory of each task and following the README there.

## License
The License will be updated after the code is released without anonymity.

## Acknowledgement
This work is built on three different repository, [RepDistiller](https://github.com/HobbitLong/RepDistiller)(ICLR 2020), [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill)(ICPR 2020) and [OverHaul](https://github.com/clovaai/overhaul-distillation)(ICCV 2019). Thanks to their great work.

