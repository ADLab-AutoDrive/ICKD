#! /bin/bash

/opt/conda/bin/python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
	--distill ickd \
	--model_s resnet8x4 \
	-a 0 -b 2.5 --trial 1
