import glob
import os
import cv2
import numpy as np
from tqdm import tqdm


def npz():
    # 图像路径
    path = r"D:/Code/DL/data/glassesB2/images/temple/train/*.png"
    # 项目中存放训练所用的npz文件路径
    path_save = r"D:/Code/DL/data/glassesB2/npz/temple/train/"
    try:
        os.makedirs(path_save)
    except FileExistsError:
        pass
    loop = tqdm(glob.glob(path))
    print(len(loop))
    for i, img_path in enumerate(loop):
        loop.set_description(f"Processing {i+1}th image: {img_path}")
        # 读入图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 读入标签
        label_path = img_path.replace("images", "labels")
        label = cv2.imread(label_path, flags=0)
        # 保存npz
        np.savez(path_save + str(i), image=image, label=label)

    # 加载npz文件
    # data = np.load(r'G:\dataset\Unet\Swin-Unet-ori\data\Synapse\train_npz\0.npz', allow_pickle=True)
    # image, label = data['image'], data['label']


if __name__ == "__main__":
    npz()
