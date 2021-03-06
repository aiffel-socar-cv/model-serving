# Model Serving using TorchServe


![torchserve](_asset/TorchServe.png)

- [x] Inference test on local
- [x] Run TorchServe server on local
    - [x] `mar_builder.py`
    - [x] `config.properties`
    - [x] manifest
    - [x] `custom handlers`
         - dent
         - scartch
         - spacing
- [x] GCS upload test 

## Payload Example
![](_asset/payload.png)

## Dependencies
- pytorch
- torchvision
- captum
- [Java 11](https://www.oracle.com/java/technologies/downloads/#java11)
- TorchServe

## Install TorchServe and Its Dependecies
Install Pytorch and TorchVision
```
conda install pytorch torchvision torchaudio -c pytorch
```
Captum(Model Interpretability for PyTorch)
```
conda install captum -c pytorch
```
Install Java 11
```
sudo apt install openjdk-11-jdk
```
Install torchserve, torch-model-archiver and torch-workflow-archiver
```
conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch
```




## How to Run the Code

**Run Server**
```
$ cd serve
$ sh start_torchserve.sh
```

**Stop Server**
```
$ torchserve --stop
```

## References
- [TORCHSERVE](https://pytorch.org/serve/)
- [Torchserve - config.properties](https://github.com/pytorch/serve/blob/master/docs/configuration.md#allow-model-specific-custom-python-packages)
- [TorchServe on Google Kubernetes Engine (GKE)](https://github.com/pytorch/serve/tree/master/kubernetes/GKE)
- [mmSegmentation Tools](https://github.com/open-mmlab/mmsegmentation/tree/master/tools/torchserve)
- [BYEONGJO's RESEARCH BLOG](https://byeongjokim.github.io/posts/MLOps-Toy-Project-0/)
- [Python requests module](https://dgkim5360.tistory.com/entry/python-requests)

### Python
- [파이썬 파일 다루기 - 임시 파일과 디렉토리](https://m.blog.naver.com/hankrah/221831304924)
