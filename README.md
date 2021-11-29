# Model Serving TEST
- [x] local에서 한 장씩 inference
- [x] local에서 torchserve 
    - [x] `mar_builder.py`
    - [x] `config.properties`
    - [x] manifest
    - [x] `custom handlers`
         - dent
         - scartch
         - spacing
- [ ] kubernetes (블로그 참조)
- [ ] GCP에 올리기 (OAuth)
- [ ] 배포 자동화 코드
- [ ] 재학습
- [ ] update

## 참고 자료
- [TORCHSERVE](https://pytorch.org/serve/)
    - [config.properties](https://github.com/pytorch/serve/blob/master/docs/configuration.md#allow-model-specific-custom-python-packages)
- [mmSegmentation Tools](https://github.com/open-mmlab/mmsegmentation/tree/master/tools/torchserve)
- [BYEONGJO's RESEARCH BLOG](https://byeongjokim.github.io/posts/MLOps-Toy-Project-0/)
- [Python requests module](https://dgkim5360.tistory.com/entry/python-requests)