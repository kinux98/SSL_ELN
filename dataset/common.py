from collections import namedtuple

import torch
import torch.nn.functional as F
from PIL import Image

import dataset.functional as Fu

# Base directories
base_voc = "./dataset/pascal_voc_seg/VOCdevkit/VOC2012"
base_city = "./dataset/cityscapes/"

# Common parameters
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

num_classes_voc = 21
colors_voc = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
    [255, 255, 255],
]
categories_voc = [
    "Background",
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Diningtable",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Pottedplant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
]

num_classes_city = 19
colors_city = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0],
]

categories_city = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
label_id_map_city = [
    255,
    255,
    255,
    255,
    255,
    255,
    255,
    0,
    1,
    255,
    255,
    2,
    3,
    4,
    255,
    255,
    255,
    5,
    255,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    255,
    255,
    16,
    17,
    18,
]
train_cities = [
    "aachen",
    "bremen",
    "darmstadt",
    "erfurt",
    "hanover",
    "krefeld",
    "strasbourg",
    "tubingen",
    "weimar",
    "bochum",
    "cologne",
    "dusseldorf",
    "hamburg",
    "jena",
    "monchengladbach",
    "stuttgart",
    "ulm",
    "zurich",
]

CityscapesClass = namedtuple(
                "CityscapesClass",
                [
                    "name",
                    "id",
                    "train_id",
                    "category",
                    "category_id",
                    "has_instances",
                    "ignore_in_eval",
                    "color",
                ],
            )

city_classes = [
                CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
                CityscapesClass(
                    "ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)
                ),
                CityscapesClass(
                    "rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)
                ),
                CityscapesClass(
                    "out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)
                ),
                CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
                CityscapesClass(
                    "dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)
                ),
                CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
                CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
                CityscapesClass(
                    "sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)
                ),
                CityscapesClass(
                    "parking", 9, 255, "flat", 1, False, True, (250, 170, 160)
                ),
                CityscapesClass(
                    "rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)
                ),
                CityscapesClass(
                    "building", 11, 2, "construction", 2, False, False, (70, 70, 70)
                ),
                CityscapesClass(
                    "wall", 12, 3, "construction", 2, False, False, (102, 102, 156)
                ),
                CityscapesClass(
                    "fence", 13, 4, "construction", 2, False, False, (190, 153, 153)
                ),
                CityscapesClass(
                    "guard rail",
                    14,
                    255,
                    "construction",
                    2,
                    False,
                    True,
                    (180, 165, 180),
                ),
                CityscapesClass(
                    "bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)
                ),
                CityscapesClass(
                    "tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)
                ),
                CityscapesClass(
                    "pole", 17, 5, "object", 3, False, False, (153, 153, 153)
                ),
                CityscapesClass(
                    "polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)
                ),
                CityscapesClass(
                    "traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)
                ),
                CityscapesClass(
                    "traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)
                ),
                CityscapesClass(
                    "vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)
                ),
                CityscapesClass(
                    "terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)
                ),
                CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
                CityscapesClass(
                    "person", 24, 11, "human", 6, True, False, (220, 20, 60)
                ),
                CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
                CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
                CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
                CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
                CityscapesClass(
                    "caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)
                ),
                CityscapesClass(
                    "trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)
                ),
                CityscapesClass(
                    "train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)
                ),
                CityscapesClass(
                    "motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)
                ),
                CityscapesClass(
                    "bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)
                ),
                CityscapesClass(
                    "license plate", -1, 255, "vehicle", 7, False, True, (0, 0, 142)
                ),
            ]


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b, candid=None):  # a for label, b for cls_output
        a = a.detach()
        b = b.detach()
        b = F.interpolate(b, size=a.size()[1:], mode="bilinear", align_corners=True)

        if candid is not None:
            candid = Fu.resize(candid, size=b.shape[2:], interpolation=Image.NEAREST)
            candid = candid.flatten()

        a = a.flatten()
        b = b.argmax(1).flatten()

        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # For pseudo labels(which has 255), just don't let your network predict 255
            if candid is not None:
                k = (a >= 0) & (a < n) & (candid == 1)
            else:
                k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu
