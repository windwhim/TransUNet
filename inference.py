import argparse
import os
import time
import cv2
import numpy as np
import torch
from scipy.ndimage import zoom
from tqdm import tqdm
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_classes", type=int, default=2, help="output channel of network"
)
parser.add_argument(
    "--img_size", type=int, default=224, help="input patch size of network input"
)
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument(
    "--n_skip", type=int, default=3, help="using number of skip-connect, default is num"
)
parser.add_argument(
    "--vit_name", type=str, default="R50-ViT-B_16", help="select one vit model"
)
parser.add_argument(
    "--vit_patches_size", type=int, default=16, help="vit_patches_size, default is 16"
)
args = parser.parse_args()


def inference(net, image: np.ndarray):
    original_size = image.shape[:2]

    image = zoom(
        image,
        (args.img_size / image.shape[0], args.img_size / image.shape[1], 1),
        order=3,
    )

    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)

    net.eval()
    with torch.no_grad():
        pred = net(image)
        out = torch.argmax(torch.softmax(pred, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()

    prediction = zoom(
        prediction,
        (
            original_size[0] / prediction.shape[0],
            original_size[1] / prediction.shape[1],
        ),
        order=5,
    )
    return prediction.astype(np.uint8) * 255


def get_files(path, ext=".png"):
    fs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                fs.append(os.path.join(root, file))
    return fs


def main():
    args.img_size = 512
    args.num_classes = 2

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    checkpoint_path = "epoch_17.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(checkpoint)

    files = get_files("./img/image")
    loop = tqdm(files, leave=True, position=0)
    for fl in loop:
        filename = os.path.basename(fl)
        loop.set_description(f"Inferencing {filename}")

        image = cv2.imread(fl)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = inference(net, image)

        cv2.imwrite(os.path.join("./img/mask", filename), mask)


if __name__ == "__main__":
    start_time = time.time()
    print(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    main()
    print(f"end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"cost time:{time.time() - start_time}")
