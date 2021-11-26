import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import albumentations as A
from efficientunet import *
from dataset import *

# from config import *
from utils import *
from metric import iou_score

INFER_DIR = "/Users/tseo/Documents/Github/viai-serving/RESULT"


def eval_model(test_loader, test_batch_num, net, criterion, optim, ckpt_dir, w_config):
    # Load Checkpoint File
    if os.listdir(ckpt_dir):
        net, optim, ckpt_path = load_net(ckpt_dir=ckpt_dir, net=net, optim=optim)

    result_dir = os.path.join(INFER_DIR, "test_" + ckpt_path.split("/")[-1][:-4])

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Evaluation
    with torch.no_grad():
        net.eval()  # Evaluation Mode
        loss_arr, iou_arr = [], []

        for batch_idx, data in enumerate(test_loader, 1):
            # Forward Propagation
            img = data["img"].to(device)
            label = data["label"].to(device)

            label = label // 255

            output = net(img)
            output_t = torch.argmax(output, dim=1).float()

            # Calc Loss Function
            loss = criterion(output, label)
            iou = iou_score(output_t, label)

            loss_arr.append(loss.item())
            iou_arr.append(iou.item())

            print_form = "[Test] | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f} | IoU: {:.4f}"
            print(print_form.format(batch_idx, test_batch_num, loss_arr[-1], iou))

            img = to_numpy(denormalization(img, mean=0.5, std=0.5))
            # 이미지 캐스팅
            img = np.clip(img, 0, 1)

            label = to_numpy(label)
            output_t = to_numpy(classify_class(output_t))

            for j in range(label.shape[0]):
                crt_id = int(w_config.BATCH_SIZE * (batch_idx - 1) + j)

                plt.imsave(os.path.join(result_dir, f"img_{crt_id:04}.png"), img[j].squeeze(), cmap="gray")
                plt.imsave(os.path.join(result_dir, f"label_{crt_id:04}.png"), label[j].squeeze(), cmap="gray")
                plt.imsave(os.path.join(result_dir, f"output_{crt_id:04}.png"), output_t[j].squeeze(), cmap="gray")

    eval_loss_avg = np.mean(loss_arr)
    eval_iou_avg = np.mean(iou_arr)
    print_form = "[Result] | Avg Loss: {:0.4f} | Avg IoU: {:0.4f}"
    print(print_form.format(eval_loss_avg, eval_iou_avg))


if __name__ == "__main__":
    from losses import FocalLoss

    TEST_IMAGES_DIR = "/Users/tseo/Documents/Github/viai-serving/DATA/dent/test/images"
    TEST_LABELS_DIR = "/Users/tseo/Documents/Github/viai-serving/DATA/dent/test/masks"

    test_transform = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485), std=(0.229)),
            transforms.ToTensorV2(),
        ]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    class Config:
        LEARNING_RATE = 1e-3
        BATCH_SIZE = 16
        NUM_EPOCHS = 100

    w_config = Config()
    batch_size = w_config.BATCH_SIZE

    test_dataset = DatasetV2(imgs_dir=TEST_IMAGES_DIR, mask_dir=TEST_LABELS_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_data_num = len(test_dataset)
    test_batch_num = int(np.ceil(test_data_num / batch_size))
    net = get_stanford_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True).to(device)
    criterion = FocalLoss(gamma=2, alpha=0.5).to(device)
    optimizer_ft = torch.optim.Adam(params=net.parameters(), lr=w_config.LEARNING_RATE)
    ckpt_dir = "/Users/tseo/Documents/Github/viai-serving/checkpoints/dent"
    eval_model(test_loader, test_batch_num, net, criterion, optimizer_ft, ckpt_dir, w_config=w_config)
