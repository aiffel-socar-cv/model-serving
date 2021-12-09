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
    def get_conf_score(output, mask):
        output = output.squeeze()
        print(output)
        print(output.shape)
        m = torch.nn.Softmax(dim=0)
        print("=" * 50)
        output = m(output)
        print(output)

        mask = mask.squeeze()
        mask_num = int(sum(mask[mask == 1]))

        if mask_num == 0:
            res = output[1]
            res = float(sum(res[res != 0]))
            n = np.count_nonzero(np.asarray(res))
            print("=" * 50)
            print(res)
            print(n)

            return res / n

        print(np.max(np.asarray(output[1])))
        print(np.min(np.asarray(output[1])))

        print(np.max(np.asarray(mask)))
        print(np.min(np.asarray(mask)))
        res = output[1] * mask
        print(np.max(np.asarray(res)))
        print(np.min(np.asarray(res)))
        print(np.count_nonzero(np.asarray(res != 0)))
        res = float(sum(res[res != 0]))
        print("=" * 30)
        print(res)
        print(mask_num)
        return res / mask_num

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
        output_t = torch.argmax(output, dim=1).float()

        conf_score = get_conf_score(output, output_t)
        output_t = to_numpy(classify_class(output_t))

    return output_t, conf_score


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
        img_data = base64.b64decode(image)
        dataBytesIO = io.BytesIO(img_data)
        image = Image.open(dataBytesIO).convert("RGB")
        image = np.asarray(image)

        # inference
        transformed = self.transform(image=image)
        np_image = transformed["image"]
        result, conf_score = inference(self.checkpoint_file, self.backbone_net, self.device, image=np_image)

        return [{"class": "dent", "mask": result.squeeze().tolist(), "confidence": conf_score}]
