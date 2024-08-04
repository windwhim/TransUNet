import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_flip(image, label):
    image = np.flip(image, axis=1).copy()
    label = np.flip(label, axis=1).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-10, 10)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, _ = image.shape

        if x != self.output_size[0] or y != self.output_size[1]:
            # MODIFY: modify zoom to resize?
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3
            )  # why not 3?
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )
            # image = cv2.resize(image, (self.output_size[1], self.output_size[0]))
            # label = cv2.resize(label, (self.output_size[1], self.output_size[0]))
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {"image": image, "label": label.long()}
        return sample


class GlassesB2(Dataset):
    def __init__(
        self, base_dir, list_dir, split, transform=None, output_size=(224, 224)
    ):
        self.output_size = output_size
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + ".txt")).readlines()
        self.data_dir = base_dir

        self.base_transform = transforms.Compose(
            [
                transforms.Resize(self.output_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # path
        slice_name = self.sample_list[idx].strip("\n")
        if self.split == "train":
            image_path = os.path.join(self.data_dir, slice_name + ".png")
            label_path = os.path.join(
                self.data_dir.replace("images", "labels"), slice_name + ".png"
            )
        else:
            image_path = os.path.join(
                self.data_dir.replace("train", "val"), slice_name + ".png"
            )
            label_path = os.path.join(
                self.data_dir.replace("images", "labels").replace("train", "val"),
                slice_name + ".png",
            )
        # load
        image = Image.open(image_path)
        label = Image.open(label_path).convert("L")
        # transform
        if self.transform:
            image = self.transform(image)
        else:
            image = self.base_transform(image)
        label = self.base_transform(label)[0]

        sample = {
            "image": image,
            "label": label,
            "case_name": slice_name,
        }

        return sample
