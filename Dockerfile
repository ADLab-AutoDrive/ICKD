FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get install -y \
        build-essential \
	emacs \
        git \
        vim \
	gcc \
	libtinfo-dev \
	zlib1g-dev \
	cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install scipy pycocotools


