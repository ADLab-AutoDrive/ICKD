# Pascal VOC
This directory provides the evaluation code of our paper for semantic segmentaion. The code is based on [OverHaul](https://github.com/clovaai/overhaul-distillation)(ICCV 2019) codebase.  

# Benchmark Results
## ResNet18
| mxn    | mIoU(%)    | checkpoint    | 
| :---   | ---:       | ---:  | 
| 1x1    | 74.14      | resnet_patch_num1.pth.tar | 
| 4x4    | 74.97      | resnet_patch_num4.pth.tar | 
| 16x16  | 74.74      | resnet_patch_num16.pth.tar | 
| 32x32  | 75.01      | resnet_patch_num32.pth.tar | 

## MobileNetV2
| mxn    | mIoU(%)    | checkpoint    | 
| :---   | ---:       | ---:  | 
| 1x1    | 72.10      | mobilenet_patch_num1.pth.tar | 
| 4x4    | 72.26      | mobilenet_patch_num1.pth.tar | 
| 16x16  | 72.79      | mobilenet_patch_num1.pth.tar | 
| 32x32  | 72.58      | mobilenet_patch_num1.pth.tar | 

## Validation
1. update the pascal voc sbd path in ${PROJECT_ROOT_DIR}/Segementation/mypath.py (Line8 & Line 10 both).
2. run eval script with the checkpoint provided and the corresponding network architecture. For example,
```   
python eval.py --backbone mobilenet --gpu-ids 0 --dataset pascal --use-sbd --checkpoint_path path_to_ckpt 
```

## Acknowledgement
[OverHaul](https://github.com/clovaai/overhaul-distillation)


   