import base64
import os
import io
import torch
import numpy as np
import albumentations as A
import cv2

import matplotlib.pyplot as plt
from ts.torch_handler.base_handler import BaseHandler
from glob import glob
from PIL import Image
from albumentations.pytorch import transforms

from efficientunet import *
from ts.torch_handler.base_handler import BaseHandler


def inference(ckpt, net, device, image):
    def get_conf_score(x):
        n = (x > 0.5).sum()
        s = x[x > 0.5].sum()
        return s / n

    def classify_class(x):
        return 1.0 * (x > 0.5)

    def to_numpy(tensor):
        if tensor.ndim == 3:
            return tensor.to("cpu").detach().numpy()
        return tensor.to("cpu").detach().numpy().transpose(0, 2, 3, 1)  # (Batch, H, W, C)

    # load model
    model_dict = torch.load(ckpt, map_location=torch.device("cpu"))  # CPU
    net.load_state_dict(model_dict["net"])

    with torch.no_grad():
        net.eval()

        image = image.to(device)
        image = image.unsqueeze(1)
        image = np.transpose(image, (1, 0, 2, 3))

        output = net(image)
        # print(output)
        # print(to_numpy(output).shape)
        output_t = torch.argmax(output, dim=1).float()
        # print("=" * 30)
        # print(output_t)
        # temp = to_numpy(output_t)
        # print(temp.shape)
        # print(np.max(temp))
        # print(np.min(temp))
        output_t = to_numpy(classify_class(output_t))

        # conf_score = get_conf_score(result_np)

    return output_t


class DentHandler(BaseHandler):
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    transform = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485), std=(0.229)),
            transforms.ToTensorV2(),
        ]
    )

    def initialize(self, context):
        self.checkpoint_dir = os.getcwd()
        self.device = "cpu"

        # load backbone network
        self.backbone_net = get_stanford_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True).to(
            self.device
        )

        # load checkpoint
        self.checkpoint_list = glob(os.path.join(self.checkpoint_dir, "*.pth"))
        assert self.checkpoint_list, self.RED + "The checkpoint file doesn't exist!!" + self.RESET
        if self.checkpoint_list:
            self.checkpoint_file = os.path.join(self.checkpoint_dir, self.checkpoint_list[0])
            print("NET: ", self.GREEN, self.checkpoint_file, self.RESET)

        self.initialized = True

    def handle(self, data, context):
        # load image
        image = data[0].get("data") or data[0].get("body")
        image = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))

        # inference
        transformed = self.transform(image=image)
        np_image = transformed["image"]
        result = inference(self.checkpoint_file, self.backbone_net, self.device, image=np_image)

        # _, buffer = cv2.imencode(".png", result.squeeze().astype("uint8") * 255)
        # content = buffer.tobytes()

        # TODO: 1. confidence score
        # TODO: 2. handle exceptions
        # return [content]

        ret = [{"class": "dent", "mask": result.squeeze().tolist()}]
        return ret
