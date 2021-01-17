import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from models.quant_layer import QuantConv2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', ]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, evaluate=False):
    """3x3 convolution with padding"""
    return QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, evaluate=evaluate)


def conv1x1(in_planes, out_planes, stride=1, evaluate=False):
    """1x1 convolution"""
    return QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, evaluate=evaluate)

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, evaluate=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, evaluate=evaluate)
        self.bn10 = norm_layer(planes)
        self.bn11 = norm_layer(planes)
        self.bn12 = norm_layer(planes)
        self.bn13 = norm_layer(planes)
        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, evaluate=evaluate)
        self.bn20 = norm_layer(planes)
        self.bn21 = norm_layer(planes)
        self.bn22 = norm_layer(planes)
        self.bn23 = norm_layer(planes)       
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, b):
        identity = x

        out = self.conv1(x, b)
        if (b==2):
            out = self.bn10(out)
        elif (b==3):
            out = self.bn11(out)
        elif (b==4):
            out = self.bn12(out)
        elif (b==5):
            out = self.bn13(out)
            
        out = self.relu(out)
        out = self.conv2(out, b)
        if (b==2):
            out = self.bn20(out)
        elif (b==3):
            out = self.bn21(out)
        elif (b==4):
            out = self.bn22(out)
        elif (b==5):
            out = self.bn23(out)
            

        if self.downsample is not None:
            identity = self.downsample(x, b)

        out += identity
        out = self.relu(out)

        return out, b


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, evaluate=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, evaluate=evaluate)
        self.bn10 = norm_layer(width)
        self.bn11 = norm_layer(width)
        self.bn12 = norm_layer(width)
        self.bn13 = norm_layer(width)
        
        self.conv2 = conv3x3(width, width, stride, groups, dilation, evaluate=evaluate)
        self.bn20 = norm_layer(width)
        self.bn21 = norm_layer(width)
        self.bn22 = norm_layer(width)
        self.bn23 = norm_layer(width)
        
        self.conv3 = conv1x1(width, planes * self.expansion, evaluate=evaluate)
        self.bn30 = norm_layer(planes * self.expansion)
        self.bn31 = norm_layer(planes * self.expansion)
        self.bn32 = norm_layer(planes * self.expansion)
        self.bn33 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, b):
        identity = x

        out = self.conv1(x, b)
        if (b==2):
            out = self.bn10(out)
        elif (b==3):
            out = self.bn11(out)
        elif (b==4):
            out = self.bn12(out)
        elif (b==5):
            out = self.bn13(out)
        out = self.relu(out)
        out = self.conv2(out, b)
        if (b==2):
            out = self.bn20(out)
        elif (b==3):
            out = self.bn21(out)
        elif (b==4):
            out = self.bn22(out)
        elif (b==5):
            out = self.bn23(out)
            
        out = self.relu(out)
        out = self.conv3(out, b)
        if (b==2):
            out = self.bn30(out)
        elif (b==3):
            out = self.bn31(out)
        elif (b==4):
            out = self.bn32(out)
        elif (b==5):
            out = self.bn33(out)
            
        if self.downsample is not None:
            identity = self.downsample(x, b)

        out += identity
        out = self.relu(out)

        return out, b


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, weight_bit=-1, data_bit=-1, evaluate=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.evaluate = evaluate
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn10 = norm_layer(self.inplanes)
        self.bn11 = norm_layer(self.inplanes)
        self.bn12 = norm_layer(self.inplanes)
        self.bn13 = norm_layer(self.inplanes)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], evaluate=evaluate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], evaluate=evaluate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], evaluate=evaluate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], evaluate=evaluate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, evaluate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            layer_subsample = [conv1x1(self.inplanes, planes * block.expansion, stride, evaluate=evaluate),
                norm_layer(planes * block.expansion)]
            downsample = mySequential(*layer_subsample)
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, evaluate=evaluate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, evaluate=evaluate))

        return mySequential(*layers)

    def forward(self, x, b):
        x = self.conv1(x)
        if (b==2):
            x = self.bn10(x)
        elif (b==3):
            x = self.bn11(x)
        elif (b==4):
            x = self.bn12(x)
        elif (b==5):
            x = self.bn13(x)
            
        x = self.relu(x)
        x = self.maxpool(x)

        x, b = self.layer1(x, b)
        x, b = self.layer2(x, b)
        x, b = self.layer3(x, b)
        x, b = self.layer4(x, b)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()



def _resnet(arch, block, layers, pretrained, progress, weight_bit=-1, data_bit=-1, evaluate=False, **kwargs):
    model = ResNet(block, layers, weight_bit=weight_bit, data_bit=data_bit, evaluate=evaluate, **kwargs)
    if pretrained:
        state_dict = load_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, weight_bit=-1, data_bit=-1, evaluate=False, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, weight_bit, data_bit, evaluate,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
