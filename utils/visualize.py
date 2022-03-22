from typing import Union, Optional, List, Tuple, Text, BinaryIO
from torchvision.utils import make_grid
import numpy as np
import cv2
import torch
from PIL import Image
import torch.nn.functional as F
import wandb
import torch.nn as nn
import copy
from dataset.common import imagenet_mean, imagenet_std

colors_voc_origin = torch.Tensor(
    [
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
        [255, 255, 255],
    ]
)


def renormalize_float(vector, range_t: tuple):

    row = torch.Tensor(vector)
    r = torch.max(row) - torch.min(row)
    row_0_to_1 = (row - torch.min(row)) / r
    r2 = range_t[1] - range_t[0]
    row_normed = (row_0_to_1 * r2) + range_t[0]

    return row_normed.numpy()


def un_normalize(img, mean=imagenet_mean, std=imagenet_std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img


def visualize_feature(
    final_candidate, label_indices, mean, std, tag, image, num_classes=20
):  # vis feature by channel
    # image : B x C x H' x W'
    # final_candidate : B x H x W (0 or 1)
    # label_indices : B x H x W (0 ~ num_classes-1)
    label_indices = copy.deepcopy(label_indices)

    image = F.interpolate(
        image.float(), size=final_candidate.size()[1:], mode="bilinear"
    )
    features = torch.where(final_candidate == 1, label_indices, 22)
    features = F.interpolate(
        features.unsqueeze(1).float(), size=image.size()[2:], mode="nearest"
    )
    features = features.detach().cpu()

    # image_origin = un_normalize(image, mean, std).detach().cpu()

    for idx in range(features.shape[0]):
        compound_score = features[idx].squeeze()
        X = compound_score  # h x w
        X = colors_voc_origin[X.long()].numpy()  # h x w x 3

        # img = image_origin[idx].numpy().squeeze()
        # img = (255*(img - np.min(img))/np.ptp(img)).astype(np.uint8)
        # img = np.transpose(img, (1,2,0))

        new_im = Image.fromarray(X.astype(np.uint8))
        # img = Image.fromarray(img.astype(np.uint8))
        # blend_image = np.asarray(Image.blend(img, new_im, alpha=0.8))

        wandb.log({str(tag) + "_" + str(idx): [wandb.Image(new_im)]}, commit=False)


def visualize_map(
    features, mean, std, tag, image, multi_channel=False
):  # vis feature by channel
    # image : B x C x H' x W'
    if multi_channel:
        # features : B x C x H x W
        features = torch.max(features, dim=1, keepdim=True)[0]
        features = F.interpolate(features, size=image.size()[2:], mode="bilinear")
    else:
        # features : B x H x W
        features = F.interpolate(
            features.unsqueeze(1).float(), size=image.size()[2:], mode="bilinear"
        )

    image_origin = image
    for idx in range(features.shape[0]):
        compound_score = features[idx].squeeze().unsqueeze(0).detach().cpu()
        X = compound_score.numpy()  # 1 x h x w
        X = np.transpose(X, (1, 2, 0))  # h x w x 1
        image = image_origin[idx]
        # X.shape : H x W
        X_image = (255 * (X - np.min(X)) / np.ptp(X)).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(X_image, cv2.COLORMAP_JET)
        image = un_normalize(image, mean, std).detach().cpu().numpy().squeeze()
        image = (255 * (image - np.min(image)) / np.ptp(image)).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        # print("heatmap/image : ",heatmap_img.shape, image.shape)
        heatmap_img = cv2.addWeighted(heatmap_img, 0.8, image, 0.2, 0)
        wandb.log(
            {"featuremap/" + str(tag) + "_" + str(idx): [wandb.Image(heatmap_img)]},
            commit=False,
        )
        if multi_channel is not True:
            pos_neg_map = compound_score.squeeze().numpy().astype(np.uint8)  #  h x w

            pallete = np.ones(
                (pos_neg_map.shape[0], pos_neg_map.shape[1], 3)
            )  # h x w x 3
            pallete[pos_neg_map == 255] = color_mappings["pos"]
            pallete[pos_neg_map == 128] = color_mappings["neg"]
            pallete[pos_neg_map == 0] = color_mappings["neut"]
            pallete = pallete.astype(np.uint8)
            heatmap_img = cv2.addWeighted(pallete, 0.8, image, 0.2, 0)
            wandb.log(
                {
                    "featuremap/pos_neg_map/"
                    + str(tag)
                    + "_"
                    + str(idx): [wandb.Image(heatmap_img)]
                },
                commit=False,
            )


def visualize_rescale_image(mean, std, image, tag):  # vis image itself with mean train
    # features : B x C x H x W
    origin_image = un_normalize(image, mean, std).detach().cpu()
    for batch_idx in range(origin_image.shape[0]):
        img = origin_image[batch_idx]
        X = img.numpy().squeeze()

        # Normalised [0,255] as integer: don't forget the parenthesis before astype(int)
        original_image = (255 * (X - np.min(X)) / np.ptp(X)).astype(np.uint8)

        # print("original image shape : ", original_image.shape)
        wandb.log(
            {
                str(tag)
                + "_"
                + str(batch_idx): [wandb.Image(np.transpose(original_image, (1, 2, 0)))]
            },
            commit=False,
        )


def visualize_stacked_feature(feature, image, mean, std, tag):
    # feature : b x c x h x w
    # image : b x c x h x w
    image_origin = image
    feature_origin = feature

    for batch_idx in range(image_origin.shape[0]):
        image = image_origin[batch_idx]  # c x h x w
        image = un_normalize(image, mean, std).detach().cpu().numpy().squeeze()
        image = (255 * (image - np.min(image)) / np.ptp(image)).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))

        X = feature_origin[batch_idx]
        X = F.interpolate(
            X.squeeze().unsqueeze(0).unsqueeze(0),
            size=image_origin.shape[2:],
            mode="bilinear",
        )
        X = X.squeeze().unsqueeze(2).detach().cpu().numpy()  # h x w x 1
        X_image = (255 * (X - np.min(X)) / np.ptp(X)).astype(np.uint8)
        # X_image.shape :  H x W x 1

        heatmap_img = cv2.applyColorMap(X_image, cv2.COLORMAP_BONE)
        final = cv2.addWeighted(heatmap_img, 0.85, image, 0.15, 0)
        wandb.log(
            {"stacked_feat_" + str(tag) + "_" + str(batch_idx): [wandb.Image(final)]},
            commit=False,
        )


def visualize_entropy_image(cls_output, image, mean, std, tag):
    # feature : b x c x h x w
    # image : b x c x h x w
    cls_output = F.interpolate(
        cls_output, size=image.shape[2:], mode="bilinear", align_corners=True
    )
    image_origin = image

    p = F.softmax(cls_output, dim=1)
    entropy_batch = torch.sum(-p * F.log_softmax(cls_output, dim=1), dim=1)

    for batch_idx in range(image_origin.shape[0]):
        image = image_origin[batch_idx]  # c x h x w
        image = un_normalize(image, mean, std).detach().cpu().numpy().squeeze()
        image = (255 * (image - np.min(image)) / np.ptp(image)).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))

        X = entropy_batch[batch_idx].squeeze().unsqueeze(2).detach().cpu().numpy()
        # X_image.shape :  H x W x 1
        X_image = (255 * (X - np.min(X)) / np.ptp(X)).astype(np.uint8)
        # normalize the heatmap
        heatmap_img = cv2.applyColorMap(X_image, cv2.COLORMAP_BONE)
        # final = cv2.addWeighted(heatmap_img, 0.8, image, 0.2, 0)
        wandb.log(
            {str(tag) + str(batch_idx): [wandb.Image(heatmap_img)]}
        )  # , commit=False)


def visualize_segmap(image, seg_map, mean, std, tag, label=False):
    # image : b x 3 x h x w with normalized
    # segmap : b x h x w
    seg_map = seg_map.detach().cpu()
    seg_map[seg_map == 255] = 21
    for batch_idx in range(image.shape[0]):
        if label is False:
            target = seg_map[batch_idx].argmax(0).squeeze()
        else:
            target = seg_map[batch_idx].squeeze()  # H x W

        new_im = colors_voc_origin[target.long()].numpy()
        new_im = Image.fromarray(new_im.astype(np.uint8))
        wandb.log(
            {str(tag) + "_" + str(batch_idx): [wandb.Image(new_im)]}, commit=False
        )


bin_color = torch.Tensor([[0, 0, 0], [255, 255, 255]])


def visualize_binary_mask(bin_mask, tag):  # vis image itself with mean train
    # features : B x C x H x W
    bin_mask = torch.round(torch.sigmoid(bin_mask)).squeeze(1)
    seg_map = bin_mask.detach().cpu()
    # seg_map[seg_map==255] = 21
    for batch_idx in range(bin_mask.shape[0]):
        target = seg_map[batch_idx].squeeze()  # H x W

        new_im = bin_color[target.long()].numpy()
        new_im = Image.fromarray(new_im.astype(np.uint8))
        wandb.log(
            {str(tag) + "_" + str(batch_idx): [wandb.Image(new_im)]}, commit=False
        )
