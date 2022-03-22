import os
from typing import Union

import numpy as np
import torchvision
import torchvision.transforms as tr
from PIL import Image

from dataset.transforms import Normalize

from .common import train_cities, city_classes


def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)


def sequence_to_string(seq: np.ndarray) -> str:
    return "".join([chr(c) for c in seq])


def pack_sequences(seqs: Union[np.ndarray, list]):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets


def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]


class dataloader(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root,
        image_set,
        transforms=None,
        transform=None,
        target_transform=None,
        label_state=True,
        data_set="voc",
        mask_type=".png",
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.mask_type = mask_type
        self.data_set = data_set

        if data_set == "voc":
            self._voc_init(root, image_set)
        elif data_set == "city":
            self._city_init(root, image_set)
            self.id_to_train_id = np.array([c.train_id for c in city_classes])
        else:
            raise ValueError

        self.ignore_index = 255

        # Different label states
        self.has_label = label_state
        self.color_jitter = tr.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.to_gray = tr.RandomGrayscale(p=0.2)

    def encode_target(self, target):
        return self.id_to_train_id[np.array(target)].astype(np.uint8)

    def get_image_and_target(
        self, index, image_v, image_o, masks_v, masks_o, total_length
    ):
        path = sequence_to_string(unpack_sequence(image_v, image_o, index))
        img = Image.open(path).convert("RGB")
        target_path = sequence_to_string(unpack_sequence(masks_v, masks_o, index))
        target = Image.open(target_path)

        return img, target

    def __getitem__(self, index):
        img, target = self.get_image_and_target(
            index,
            self.image_v,
            self.image_o,
            self.masks_v,
            self.masks_o,
            self.image_len,
        )

        if self.data_set == "city":
            target = self.encode_target(target)
            target = Image.fromarray(target)

        if self.has_label:
            img, target = self.transforms(img, target)
            return img, target
        else:
            img, target = self.transforms(img, target)
            img_aug = self.to_gray(self.color_jitter(img))
            img, target = Normalize()(img, target)
            img_aug = Normalize()(img_aug)
            return img, target, img_aug

    def __len__(self):
        return self.image_len

    def _voc_init(self, root, image_set):
        image_dir = os.path.join(root, "JPEGImages")

        mask_dir = os.path.join(
            root, "SegmentationClassAug"
        )  # os.path.join(root, 'SegmentationClassAugPseudo')

        splits_dir = os.path.join(root, "ImageSets/Segmentation")
        split_f = os.path.join(splits_dir, image_set + ".txt")
        print(split_f)

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        ################# 100% #################
        images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        print("labeled 100% : ", len(images))
        self.image_len = len(images)
        img_seq = [string_to_sequence(s) for s in images]
        self.image_v, self.image_o = pack_sequences(img_seq)

        masks = [os.path.join(mask_dir, x + self.mask_type) for x in file_names]
        mask_seq = [string_to_sequence(s) for s in masks]
        self.masks_v, self.masks_o = pack_sequences(mask_seq)
        #########################################

    def _city_init(self, root, image_set):
        # It's tricky, lots of folders
        image_dir = os.path.join(root, "leftImg8bit")
        check_dirs = False
        mask_dir = os.path.join(root, "gtFine")

        if image_set == "val" or image_set == "test":
            image_dir = os.path.join(image_dir, image_set)
            mask_dir = os.path.join(mask_dir, image_set)
        else:
            image_dir = os.path.join(image_dir, "train")
            mask_dir = os.path.join(mask_dir, "train")

        if check_dirs:
            for city in train_cities:
                temp = os.path.join(mask_dir, city)
                if not os.path.exists(temp):
                    os.makedirs(temp)

        # We first generate data lists before all this, so we can do this easier
        splits_dir = os.path.join(root, "data_lists")
        split_f = os.path.join(splits_dir, image_set + ".txt")
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        print(split_f)
        ################# 100% #################
        images = [os.path.join(image_dir, x + "_leftImg8bit.png") for x in file_names]
        print("labeled 100% : ", len(images))
        self.image_len = len(images)
        img_seq = [string_to_sequence(s) for s in images]
        self.image_v, self.image_o = pack_sequences(img_seq)

        masks = [
            os.path.join(mask_dir, x + "_gtFine_labelIds" + self.mask_type)
            for x in file_names
        ]
        mask_seq = [string_to_sequence(s) for s in masks]
        self.masks_v, self.masks_o = pack_sequences(mask_seq)
        #########################################
