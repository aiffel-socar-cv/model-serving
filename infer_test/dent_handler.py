import base64
import os
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from ts.torch_handler.base_handler import BaseHandler
from glob import glob
from PIL import Image
from albumentations.pytorch import transforms

from efficientunet import *


class DentHandler(BaseHandler):
    transform = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485), std=(0.229)),
            transforms.ToTensorV2(),
        ]
    )

    def initialize(self, context):
        self.device = "cpu"

        ## GPU
        # properties = context.system_properties
        # self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = torch.device(
        #     self.map_location + ":" + str(properties.get("gpu_id")) if torch.cuda.is_available() else self.map_location
        # )

        self.backbone_net = get_stanford_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True).to(device)
        self.checkpoint_dir = os.getcwd()

        self.initialized = True

    def preprocess(self, data):
        return data

    def inference(self, data, *args, **kwargs):
        return data

    def postprocess(self, data):
        # output = []

        return [data]

    def handle(self, data, context):
        print(type(data))
        print(data)

        return []
