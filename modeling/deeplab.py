import torch
import torch.nn as nn
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import resnet
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class classifier(nn.Module):
    def __init__(self, num_classes=21, out_dim=256):
        super(classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, num_classes, kernel_size=1, stride=1),
        )

    @torch.cuda.amp.autocast()
    def forward(self, x, expand=None):
        cls = self.classifier(x)
        return cls  # b x num_class


class projector(nn.Module):
    def __init__(self, out_dim=256):
        super(projector, self).__init__()

        self.projector = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, 128, kernel_size=1, stride=1),
        )

    @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.projector(x)
        return x


class DeepLab(nn.Module):
    def __init__(self, output_stride=16, resnet_name=101):
        super(DeepLab, self).__init__()
        BatchNorm = SynchronizedBatchNorm2d

        if resnet_name == 101:
            self.backbone = resnet.ResNet101(output_stride, BatchNorm)
        elif resnet_name == 50:
            self.backbone = resnet.ResNet50(output_stride, BatchNorm)

    @torch.cuda.amp.autocast()
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        return x, low_level_feat

    def get_backbone_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class Decoder(nn.Module):
    def __init__(
        self, backbone="resnet", output_stride=16, num_classes=256, num_cls=21
    ):
        BatchNorm = SynchronizedBatchNorm2d

        super(Decoder, self).__init__()
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.cls = classifier(num_classes=num_cls)
        self.proj = projector()

    @torch.cuda.amp.autocast()
    def forward(self, x, low_level_feat, gt_mode=False):
        x = self.aspp(x, gt_mode)
        x_ = self.decoder(x, low_level_feat, gt_mode)
        cls = self.cls(x_)
        proj = self.proj(x_)
        return cls, proj

    def get_other_params(self):
        modules = [self.aspp, self.decoder, self.cls, self.proj]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
