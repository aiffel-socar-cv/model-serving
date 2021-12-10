# Model Serving TEST



## HOW TO RUN TEST

**run server**
```
$ cd serve
$ sh start_torchserve.sh
```

**stop server**
```
$ torchserve --stop
```
```
$ sh stop_torchserve.sh
```

## TO DO LIST
- [x] local에서 한 장씩 inference
- [x] local에서 torchserve 
    - [x] `mar_builder.py`
    - [x] `config.properties`
    - [x] manifest
    - [x] `custom handlers`
         - dent
         - scartch
         - spacing

## 참고 자료
- [TORCHSERVE](https://pytorch.org/serve/)
- [Torchserve - config.properties](https://github.com/pytorch/serve/blob/master/docs/configuration.md#allow-model-specific-custom-python-packages)
- [TorchServe on Google Kubernetes Engine (GKE)](https://github.com/pytorch/serve/tree/master/kubernetes/GKE)
- [mmSegmentation Tools](https://github.com/open-mmlab/mmsegmentation/tree/master/tools/torchserve)
- [BYEONGJO's RESEARCH BLOG](https://byeongjokim.github.io/posts/MLOps-Toy-Project-0/)
- [Python requests module](https://dgkim5360.tistory.com/entry/python-requests)

### Python
- [파이썬 파일 다루기 - 임시 파일과 디렉토리](https://m.blog.naver.com/hankrah/221831304924)