# Imagenet
This respository provides the evaluation code of our paper, the code is based on [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) (ICPR 2020) codebase to validate our models. 

## ImageNet(ILSVRC 2012)
|                 | Vanilla    | KD    | AT    | RKD      | SCKD\*  | CRD   | CRD+KD  | SAD\*  | CC\*   | ICKD-C(Ours)\*| Teacher |   
| :---            | ---:       | ---:  | ---:  | ---:     | ---:    | ---:  | ---:    | ---:   | ---:   | ---:          | ---:    |   
| Top-1           | 70.04\*    | 70.68 | 70.59 | 71.34    | 70.87   | 71.17 | 71.38   | 71.38  | 70.74  | 72.16         | 73.31   |   
| Top-5           | 89.48      | 90.16 | 89.73 | 90.37    | N/A     | 90.13 | 90.49   | N/A    | N/A    | 90.75         | 91.42   |   
  
Top-1 and Top-5 Accuracy(%) on ImageNet validation set.

## Requirements
- Python 3.6 >=
- pipenv (optional)

### Install by pip/pipenv
```
pip3 install torchdistill
# or use pipenv
pipenv install torchdistill
```

### Install from this repository 
```
git clone https://github.com/yoshitomo-matsubara/torchdistill.git
cd torchdistill/
pip3 install -e .
# or use pipenv
pipenv install "-e ."
```

### Docker image
We provide a dockerfile to build an image.
```
docker build . -f ./dockerfile.autodrive -t ickd
```
Run it with
```
nvidia-docker run --shm-size 32G --net=host -v {DATA_DIR}:/data -it ickd
```
 
## Validation 
```
./test_local.sh
```
- Our trained resnet18 models are available to download here:
[resnet18-from-resnet34](oss://xuanyuan-div/liuli/exp/torchdistill/iccv_results/imagenet/imagenet-resnet18_from_resnet34.pt)
[resnet18-from-resnet50](oss://xuanyuan-div/liuli/exp/torchdistill/iccv_results/imagenet/imagenet-resnet18_from_resnet50.pt)
[resnet18-from-resnet101](oss://xuanyuan-div/liuli/exp/torchdistill/iccv_results/imagenet/imagenet-resnet18_from_resnet101.pt)
- Put the downloaded model to the student_model.ckpt path, which is set on `configs/sample/ilsvrc2012/single_stage/cs/test_resnet18_from_resnet34.yaml`


## Acknowledgement
The work is based on [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill)
