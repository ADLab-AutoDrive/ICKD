#! /bin/bash

NUM_GPUS=1
WORLD_SIZE=1
/opt/conda/bin/python -m torch.distributed.launch \
	--nproc_per_node=${NUM_GPUS} \
	--use_env examples/image_classification.py \
	--config configs/sample/ilsvrc2012/single_stage/cs/test_resnet18_from_resnet101.yaml \
	--log /result/ilsvrc2012/cs/resnet18_from_resnet34.txt \
	--world_size ${WORLD_SIZE} \
	-adjust_lr \
	-test_only
