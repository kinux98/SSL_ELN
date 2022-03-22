import argparse
import os
from dataset.transforms import *
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import dataset.functional as Fd
from dataset.common import imagenet_mean, imagenet_std, colors_voc, colors_city
from dataset.transforms import ToTensor
from modeling.ELN import ELNetwork
from modeling.deeplab import DeepLab, Decoder
from utils.visualize import un_normalize

def visualize_image(image): # vis image itself with mean train
    # features : B x C x H x W
    origin_image = un_normalize(image).detach().cpu()
    
    img = origin_image[0]
    X = img.numpy().squeeze()
    # Normalised [0,255] as integer: don't forget the parenthesis before astype(int)
    original_image = (255*(X - np.min(X))/np.ptp(X)).astype(np.uint8)
    #print("original image shape : ", original_image.shape)
    cv2.imwrite("./input.jpg", np.transpose(original_image, (1,2,0)))



def visualize_segmap(seg_map, dataset='voc', label=False):
    # segmap : 1 x 21 x  h x w 
    seg_map = seg_map.detach().cpu()
    
    if dataset == 'voc':
        seg_map[seg_map == 255] = 21
    elif dataset == 'city':
        seg_map[seg_map == 255] = 19

    print(seg_map.shape)

    if label is False:
        target = seg_map.argmax(1).squeeze()
    else:
        target = seg_map.squeeze()
    
    if dataset == 'voc':
        colors_voc_origin = torch.Tensor(colors_voc)
        new_im = colors_voc_origin[target.long()].numpy()
    elif dataset == 'city':
        colors_voc_origin = torch.Tensor(colors_city)
        new_im = colors_voc_origin[target.long()].numpy()

    new_im = new_im.astype(np.uint8)

    cv2.imwrite('./segmentation_'+str(int(label))+'.png', new_im)

def visualize_binary_mask(bin_mask): # vis image itself with mean train
    bin_mask = torch.round(torch.sigmoid(bin_mask.float())).squeeze(1)
    seg_map = bin_mask.detach().cpu()
    # seg_map[seg_map==255] = 21
    bin_color = torch.Tensor([[0,0,0], [255,255,255]])
    target = seg_map[0].squeeze() # H x W
    new_im = bin_color[target.long()].numpy()
    new_im = new_im.astype(np.uint8)

    cv2.imwrite('./ELN_mask.png', new_im)

def visualize_filtered_map(cls_output, bin_mask, dataset):
    cls_output = cls_output.detach().cpu()
    bin_mask = bin_mask.detach().cpu()

    bin_mask = torch.round(torch.sigmoid(bin_mask.float())).squeeze() # H x W
    cls_output = cls_output.argmax(1).squeeze() # H x W 

    if dataset == 'voc':
        filtered_output = torch.where(bin_mask == 1, cls_output, 21)
        colors_voc_origin = torch.Tensor(colors_voc)
    elif dataset == 'city':
        filtered_output = torch.where(bin_mask == 1, cls_output, 19)
        colors_voc_origin = torch.Tensor(colors_city)
        
    new_im = colors_voc_origin[filtered_output.long()].numpy()

    cv2.imwrite('./filtered_map.png', new_im)

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load image
    img = Image.open(args.image_path).convert("RGB")
    target = Image.open(args.gt_path)
    
    if args.dataset == 'voc':
        transform_train = Compose(
                [
                    ToTensor(),
                    RandomResize(min_size=(505, 505)),
                    Normalize(),
                ]
            )
    elif args.dataset == 'city':
        transform_train = Compose(
            [
                ToTensor(),
                RandomCrop(size=(512, 1024)),
                Normalize(),
            ]
        )
    img, target = transform_train(img, target)
    img_norm = img.unsqueeze(0).cuda()

    if args.dataset == 'voc':
        num_classes = 21
    elif args.dataset == 'city':
        num_classes = 19
    else:
        raise ValueError
    
    enc = DeepLab(resnet_name=args.backbone_name).cuda()
    dec = Decoder(num_cls=num_classes).cuda()
    eln = ELNetwork(num_classes=num_classes).cuda() 

    loaded_struct = torch.load(os.path.join(os.getcwd(), args.pretrained_ckpt))

    enc.load_state_dict(loaded_struct['enc_state_dict'], strict=True)
    dec.load_state_dict(loaded_struct['dec_state_dict'], strict=True)
    eln.load_state_dict(loaded_struct['eln_state_dict'], strict=True)

    with torch.no_grad():
        x4, x1 = enc(img_norm)
        cls_output, _ = dec(x4, x1)

    cls_output = F.interpolate(cls_output, size=img_norm.size()[2:], mode="bilinear", align_corners=True)
    
    visualize_image(img_norm)
    # visualize segmentation map
    visualize_segmap(cls_output, args.dataset)
    visualize_segmap(target, args.dataset, label=True)

    # visualize binary map
    final_candid_sup_ = eln(img_norm, cls_output) #self.cor(comb_input, cls_output)# b x 22 x h x w
    final_candid_sup_ = F.interpolate(final_candid_sup_, size=img_norm.shape[2:], mode='bilinear', align_corners=True)

    visualize_binary_mask(final_candid_sup_)
    visualize_filtered_map(cls_output, final_candid_sup_, args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20170890,
                        help='Random seed ')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Train/Evaluate on PASCAL VOC 2012(voc)/Cityscapes(city) (default: voc)')
    parser.add_argument('--pretrained-ckpt', type=str, default=None,
                        help='Load pretrained weight, write path to weight (default: None)')
    parser.add_argument('--backbone_name', type=int, choices=[50,101], default=101)
    parser.add_argument('--image-path', type=str, default=None,
                        help='Path of image for visualization')
    parser.add_argument('--gt-path', type=str, default=None,
                        help='Path of gt for visualization')
    
    args = parser.parse_args()
    main(args)
