import os
import torch

from glob import glob
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import transforms

# from archs.utils import classify_class, to_numpy
from efficientunet import *


import matplotlib.pyplot as plt


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"


def preprocess(image_path):
    transform = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485), std=(0.229)),
            transforms.ToTensorV2(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    np_image = np.array(image)
    transformed = transform(image=np_image)
    np_image = transformed["image"]

    return np_image


def show_image(image_path):
    transform = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485), std=(0.229)),
            # transforms.ToTensorV2(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    np_image = np.array(image)
    transformed = transform(image=np_image)
    np_image = transformed["image"]
    plt.imshow(np_image)
    plt.show()


def create_folder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print(Colors.RED, "Error: Creating directory. ", dir, Colors.RESET)


def inference(ckpt, net, device, image_path, save_dir):
    def classify_class(x):
        return 1.0 * (x > 0.5)

    def to_numpy(tensor):
        if tensor.ndim == 3:
            return tensor.to("cpu").detach().numpy()
        return tensor.to("cpu").detach().numpy().transpose(0, 2, 3, 1)  # (Batch, H, W, C)

    # if device == "cpu":
    model_dict = torch.load(ckpt, map_location=torch.device("cpu"))
    # else:  # GPU
    # model_dict = torch.load(ckpt)

    net.load_state_dict(model_dict["net"])
    # print(net)

    with torch.no_grad():
        net.eval()

        image = preprocess(image_path).to(device)
        image = image.unsqueeze(1)
        image = np.transpose(image, (1, 0, 2, 3))

        # print(image.shape)
        output = net(image)
        output_t = torch.argmax(output, dim=1).float()
        output_t = to_numpy(classify_class(output_t))

        ## TODO: add postprocess
        image_name, _ = os.path.splitext(image_path.split("/")[-1])
        # print(image_name)
        plt.imsave(os.path.join(save_dir, f"{image_name}.png"), output_t.squeeze(), cmap="gray")


if __name__ == "__main__":
    TEST_IMAGE = "/Users/tseo/Documents/Github/viai-serving/DATA/dent/test/images/20190227_10535_20402100_8621821f5ed8c37bc0b8e6e0efc9cc0a.jpg"
    CHECKPOINT_DIR = os.getcwd()  # current directory

    SAVE_DIR = "/Users/tseo/Documents/Github/viai-serving/RESULT/dent"
    create_folder(SAVE_DIR)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## load backbone networkd
    backbone_net = get_stanford_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True).to(device)

    ## load U-net
    checkpoint_list = glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    assert checkpoint_list, Colors.RED + "The checkpoint file doesn't exist!!" + Colors.RESET
    if checkpoint_list:
        CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, checkpoint_list[0])
        print("NET: ", Colors.GREEN, CHECKPOINT_FILE, Colors.RESET)

    inference(CHECKPOINT_FILE, backbone_net, device, image_path=TEST_IMAGE, save_dir=SAVE_DIR)
    # print(preprocess(TEST_IMAGE))
