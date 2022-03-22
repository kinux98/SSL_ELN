import math
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.nn.functional as F

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, output_stride, BatchNorm, pretrained=True, url=None
    ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            stride=strides[0],
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=strides[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=strides[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.layer4 = self._make_MG_unit(
            block,
            512,
            blocks=blocks,
            stride=strides[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )
        self._init_weight()

        if pretrained:
            print("load pretrained RESNET model's weight")
            self._load_pretrained_model(url)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation, downsample, BatchNorm)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm)
            )

        return nn.Sequential(*layers)

    def _make_MG_unit(
        self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                dilation=blocks[0] * dilation,
                downsample=downsample,
                BatchNorm=BatchNorm,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=blocks[i] * dilation,
                    BatchNorm=BatchNorm,
                )
            )

        return nn.Sequential(*layers)

    @torch.cuda.amp.autocast()
    def forward(self, input, intercept=False):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        if intercept is True:
            return x_1, x_3, x_4
        return x_4, x_1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, url):
        pretrain_dict = model_zoo.load_url(url)
        # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Use ResNet101 as a Backbone Network")
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        output_stride,
        BatchNorm,
        pretrained=pretrained,
        url=model_urls["resnet101"],
    )
    return model


def ResNet50(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Use ResNet50 as a Backbone Network")
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        output_stride,
        BatchNorm,
        pretrained=pretrained,
        url=model_urls["resnet50"],
    )
    return model
