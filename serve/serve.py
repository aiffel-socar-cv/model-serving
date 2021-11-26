import os
import torch

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import transforms


import matplotlib.pyplot as plt


def eval():
    pass


if __name__ == "__main__":
    TEST_IMAGE = "/Users/tseo/Documents/Github/viai-serving/DATA/dent/test/images/20190505_7417_22792212_59a1b26112a4bcbd4e964d98a14a3ae5.jpg"

    test_transform = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485), std=(0.229)),
            # transforms.ToTensorV2(),
        ]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## 이미지 불러오기
    # test_image = np.array()
    image = Image.open(TEST_IMAGE).convert("RGB")

    np_image = np.array(image)
    transformed = test_transform(image=np_image)
    np_image = transformed["image"]

    plt.imshow(np_image)
    plt.show()

    ## efficient net 불러오기
    ## U-net 체크포인트 불러오기
