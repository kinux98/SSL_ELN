# Mostly copied and modified from torch/vision/references/segmentation to support unlabeled data
# Copied functions from fmassa/vision-1 to support multi-dimensional masks loaded from numpy ndarray
import numpy as np
from PIL import Image
import random
import torch
import dataset.functional as F
import cv2
from torchvision import transforms
from dataset.common import imagenet_mean, imagenet_std
import numbers

# borrowed from https://github.com/voldemortX/DST-CBC/blob/master/segmentation/utils/transforms.py
# For 2/3 dimensional tensors only
def get_tensor_image_size(img):
    if img.dim() == 2:
        h, w = img.size()
    else:
        h = img.size()[1]
        w = img.size()[2]

    return h, w


class RandomCrop4(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        if len(img.shape) == 4:  # b x c x h x w
            h, w = img.shape[2:]
        elif len(img.shape) == 3:  # c x h x w
            h, w = img.shape[1:]
        else:
            raise NotImplementedError
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, i1, i2, i3, i4):
        # Pad if needed
        i, j, h, w = self.get_params(i1, self.size)

        i1 = F.crop(i1, i, j, h, w)
        i2 = F.crop(i2, i, j, h, w)
        i3 = F.crop(i3, i, j, h, w)
        i4 = F.crop(i4, i, j, h, w)

        return i1, i2, i3, i4


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, *args):
        for t in self.transforms:
            image, target = t(image, target)
        return (image, target, *args)


class Resize(object):
    def __init__(self, size_image, size_label):
        self.size_image = size_image
        self.size_label = size_label

    def __call__(self, image, target):
        image = (
            image
            if type(image) == str
            else F.resize(image, self.size_image, interpolation=Image.LINEAR)
        )
        target = (
            target
            if type(target) == str
            else F.resize(target, self.size_label, interpolation=Image.NEAREST)
        )

        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        min_h, min_w = self.min_size
        max_h, max_w = self.max_size
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        image = F.resize(image, (h, w), interpolation=Image.BILINEAR)
        target = (
            target
            if type(target) == str
            else F.resize(target, (h, w), interpolation=Image.NEAREST)
        )

        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        h, w = get_tensor_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, target):
        # Pad if needed
        ih, iw = get_tensor_image_size(image)
        # if ih < self.size[0] or iw < self.size[1]:
        #     image, target = ZeroPad.zero_pad(image, target,
        #                                      max(self.size[0], ih),
        #                                      max(self.size[1], iw))
        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)
        target = F.crop(target, i, j, h, w)

        return image, target


class UnNormalize(object):
    def __init__(self, mean=imagenet_mean, std=imagenet_std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class RandomJitter(object):
    def __init__(self) -> None:
        self.jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

    def __call__(self, image, target):
        if random.random() < 0.4:
            image = self.jitter(image)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        t = random.random()
        if t < self.flip_prob:
            image = F.hflip(image)

        if target is not None:
            target = (
                target
                if (type(target) == str or t >= self.flip_prob)
                else F.hflip(target)
            )
            return image, target
        else:
            return image


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        t = random.random()
        if t < self.flip_prob:
            image = F.vflip(image)

        if target is not None:
            target = (
                target
                if (type(target) == str or t >= self.flip_prob)
                else F.vflip(target)
            )
            return image, target
        else:
            return image


class ToTensor(object):
    def __init__(self, keep_scale=False, reverse_channels=False):
        # keep_scale = True => Images or whatever are not divided by 255
        # reverse_channels = True => RGB images are changed to BGR(the default behavior of openCV & Caffe,
        #                                                          let's wish them all go to heaven,
        #                                                          for they wasted me days!)
        self.keep_scale = keep_scale
        self.reverse_channels = reverse_channels

    def __call__(self, image, target):
        image = image if type(image) == str else self._pil_to_tensor(image)
        target = target if type(target) == str else self.label_to_tensor(target)

        return image, target

    @staticmethod
    def label_to_tensor(pic):  # 3 dimensional arrays or normal segmentation masks
        if isinstance(pic, np.ndarray):
            return torch.as_tensor(pic.transpose((2, 0, 1)), dtype=torch.float32)
        else:
            return torch.as_tensor(np.array(pic), dtype=torch.int64)

    def npy_to_tensor(pic):  # 3 dimensional arrays or normal segmentation masks
        return torch.as_tensor(pic, dtype=torch.float32).squeeze()

    def _pil_to_tensor(self, pic):
        # Convert a PIL Image to tensor(a direct copy)
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == "F":
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == "1":
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        if (
            self.reverse_channels
        ):  # Beware this only works with 3 channels(can't use -1 with tensors)
            img = img[:, :, [2, 1, 0]]
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            if self.keep_scale:
                return img.float()
            else:
                return img.float().div(255)
        else:
            return img


class Normalize(object):
    def __init__(self, mean=imagenet_mean, std=imagenet_std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)

        if target is None:
            return image
        return image, target


class LabelMap(object):
    def __init__(self, label_id_map):
        self.label_id_map = torch.tensor(label_id_map)

    def __call__(self, image, target):
        target = target if type(target) == str else self.label_id_map[target]

        return image, target
