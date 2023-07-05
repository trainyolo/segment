import os
import random

from torch.utils.data import Dataset
import json
import numpy as np

from PIL import Image
import random

from utils.image_utils import load_image
import yaml


class SegmentationDataset(Dataset):
    def __init__(
        self, location="./dataset", split="TRAIN", fake_size=None, transform=None
    ):
        # get dataset info
        with open(os.path.join(location, "dataset.yaml"), "r") as f:
            dataset_info = yaml.load(f, Loader=yaml.FullLoader)

        # get split
        split_f = dataset_info["train"] if split == "TRAIN" else dataset_info["val"]
        with open(os.path.join(location, split_f), "r") as f:
            data = f.read().splitlines()

        self.location = location
        self.dataset_info = dataset_info
        self.data = data
        self.size = len(data)
        self.fake_size = fake_size
        self.transform = transform

        print(f"SegmentationDataset created, found {self.size} samples")

    def __len__(self):
        return self.fake_size or self.size

    def get_index(self, index):
        if self.fake_size:
            index = random.randint(0, self.size - 1) if self.size > 1 else 0
        return index

    def get_image(self, index):
        image_filename = self.data[index]
        image_path = os.path.join(self.location, image_filename)
        image = load_image(image_path)
        return image

    def get_label(self, index):
        label_filename = self.data[index].rsplit(".", 1)[0].replace('images', 'labels', 1) + ".png"
        label_path = os.path.join(self.location, label_filename)
        label = np.array(Image.open(label_path))

        if len(self.dataset_info["names"]) > 1:
            label = label - 1

        return label

    def __getitem__(self, index):
        index = self.get_index(index)

        image = np.array(self.get_image(index))
        label = self.get_label(index)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        sample = {
            "image": image,
            "label": label,
        }

        return sample
