import base64
import os
import io
from numpy.core.shape_base import stack
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

from tempfile import TemporaryFile
from google.cloud import storage

BUCKET_NAME = "images-inferred"


def inference(ckpt, net, device, image):
    """Inffer Image
    Args:
        ckpt: trained checkpoint file
        net: backbone network
        device: cpu or gpu
        image: image to be inffered

    Return:
        output_t: inffered result (numpy array)
        conf_score: confidence score
    """

    def get_conf_score(output, mask):
        """
        get confidence score using softmax
        """
        output = output.squeeze()
        m = torch.nn.Softmax(dim=0)
        output = m(output)

        mask = mask.squeeze()
        mask_num = int(sum(mask[mask == 1]))

        if mask_num == 0:
            return 0

        res = output[1] * mask
        res = float(sum(res[res != 0]))
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
    # Error message colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    # preprocess transform
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
        """main process of the handler
        Args:
            data:
                data[0].get("name") : the name of the image (ex: AAA.jpg)
                data[0].get("body") : iamge to be inffered
            context: metadata of environment (not used)

        Returns:
            class: class type
            mask: uploaded path
            confidence: confidence score
        """
        # get image name from data
        image_name = data[0].get("name")
        image_name = image_name.decode()
        image_name, _ = os.path.splitext(image_name)

        # get image from data
        image = data[0].get("data") or data[0].get("body")

        # bytes to ndarray
        img_data = base64.b64decode(image)
        dataBytesIO = io.BytesIO(img_data)
        image = Image.open(dataBytesIO).convert("RGB")
        image = np.asarray(image)

        # inference
        transformed = self.transform(image=image)
        np_image = transformed["image"]
        result, conf_score = inference(self.checkpoint_file, self.backbone_net, self.device, image=np_image)

        # postprocess
        result = result.squeeze()
        result = result * 255  # uint8
        result = result.astype(np.uint8)
        stacked_img = np.stack((result,) * 3, axis=-1)
        im = Image.fromarray(stacked_img)

        # upload to bucket
        destination_blob_name = f"masks-dent/{image_name}.png"
        storage_client = storage.Client.from_service_account_json("./aiffel-gn-3-c8c200820331.json")
        bucket = storage_client.bucket(BUCKET_NAME)
        with TemporaryFile() as gcs_image:
            im.save(gcs_image, "png")
            gcs_image.seek(0)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_file(gcs_image)
            print(f"File {image_name}.png uploaded to {destination_blob_name}.")

        return [{"class": "dent", "mask": destination_blob_name, "confidence": conf_score}]
