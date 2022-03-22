from PIL import Image
from dataset.transforms import RandomCrop4
import torch.nn as nn
import torch
import torch.nn.functional as F


def CEloss(inputs, gt, ignore_index=255):
    inputs = F.interpolate(
        inputs, size=gt.size()[1:], mode="bilinear", align_corners=True
    )
    return nn.CrossEntropyLoss(ignore_index=ignore_index)(inputs, gt)


def total_loss(losses_list):
    total = 0
    for component in losses_list:
        if isinstance(component, list):
            total += sum(component)
        else:
            total += component
    return total


#########################################


### pixel-wise
crop_f = RandomCrop4(size=(32, 32))


def batch_pixelwise_distanceloss(self, inputs, target, final_candidate, final_indices):
    # input : b x c x h x w
    # target : b x c x h x w
    # final_indicies : b x h x w

    assert (
        inputs.size()[2:] == final_candidate.size()[1:]
    ), "input / final_candid : %s, %s" % (inputs.shape, final_candidate.shape)

    final_indices = torch.where(final_candidate == 1, final_indices, 255)

    b_loss = 0
    label = torch.unique(final_indices)
    label = label[label != 255]

    # mask : b x h x w
    inputs = inputs.permute(1, 0, 2, 3)  # c x b x h x w
    target = target.permute(1, 0, 2, 3)  # c x b x h x w

    for idx in label:
        # if idx == 0:
        #     continue
        input_vec = inputs[:, (idx == final_indices)].T  # n x 128
        input_vec = input_vec / input_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)

        pos_vec = target[:, (idx == final_indices)]  # 128 x n
        pos_vec = pos_vec / pos_vec.norm(dim=0, keepdim=True).clamp(min=1e-8)

        neg_v = target[:, (idx != final_indices) & (final_indices != 255)]  # 128 x m
        neg_v = neg_v / neg_v.norm(dim=0, keepdim=True).clamp(min=1e-8)

        pos_pair = torch.mm(input_vec, pos_vec)
        neg_pair = torch.mm(input_vec, neg_v)
        pos_pair = torch.exp(pos_pair / self.args.temp).sum().clamp(min=1e-8)
        neg_pair = torch.exp(neg_pair / self.args.temp).sum().clamp(min=1e-8)

        b_loss += -(torch.log(pos_pair / (neg_pair + pos_pair))) / torch.count_nonzero(
            (idx == final_indices).long()
        )

    if len(label) == 0:
        return 0
    else:
        return b_loss / len(label)


def pixelwisecontrastiveloss(
    self, inputs, target, final_candidate=None, final_indicies=None
):

    assert final_candidate is not None

    tot_loss = 0
    crop_cnt = 8

    for _ in range(crop_cnt):
        cr_inputs, cr_target, cr_final_candid, cr_final_indicies = crop_f(
            inputs, target, final_candidate, final_indicies
        )

        tot_loss += batch_pixelwise_distanceloss(
            self, cr_inputs, cr_target, cr_final_candid, cr_final_indicies
        )
    return tot_loss / crop_cnt
