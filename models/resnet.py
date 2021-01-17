import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from models.quant_layer import QuantConv2d

__all__ = ['ResNet', 'resnet18', 'resnet50', ]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, bit, stride=1, groups=1, dilation=1):
    """3x3 convolution with term quantization"""
    return QuantConv2d(in_planes, out_planes, bit, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, bit, stride=1):
    """1x1 convolution with term quantization"""
    return QuantConv2d(in_planes, out_planes, bit, kernel_size=1, stride=stride, bias=False)


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

    def __init__(self, inplanes, planes, bit, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, evaluate=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, bit, stride)
        self.conv2 = conv3x3(planes, planes, bit)
        
        # initialize the BN layer for each of the term budget
        self.bn10 = norm_layer(planes)
        self.bn11 = norm_layer(planes)
        self.bn12 = norm_layer(planes)
        self.bn13 = norm_layer(planes)
        self.bn14 = norm_layer(planes)
        self.bn15 = norm_layer(planes)
        self.bn16 = norm_layer(planes)
        self.bn17 = norm_layer(planes)
        self.bn18 = norm_layer(planes)
        
        self.bn20 = norm_layer(planes)
        self.bn21 = norm_layer(planes)
        self.bn22 = norm_layer(planes)
        self.bn23 = norm_layer(planes)
        self.bn24 = norm_layer(planes)
        self.bn25 = norm_layer(planes)
        self.bn26 = norm_layer(planes)
        self.bn27 = norm_layer(planes)
        self.bn28 = norm_layer(planes)
        ######################################################        
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, a, k):
        
        identity = x 
        out = self.conv1(x, a, k)
        if ((a==2) and (k==8)):
            out = self.bn10(out)
        elif ((a==2) and (k==10)):
            out = self.bn11(out)
        elif ((a==2) and (k==12)):
            out = self.bn12(out)
        elif ((a==2) and (k==14)):
            out = self.bn13(out)
        elif ((a==3) and (k==14)):
            out = self.bn14(out)
        elif ((a==3) and (k==16)):
            out = self.bn15(out)
        elif ((a==3) and (k==18)):
            out = self.bn16(out)            
        elif ((a==3) and (k==20)):
            out = self.bn17(out)

        out = self.relu(out)

        out = self.conv2(out, a, k)
        if ((a==2) and (k==8)):
            out = self.bn20(out)
        elif ((a==2) and (k==10)):
            out = self.bn21(out)
        elif ((a==2) and (k==12)):
            out = self.bn22(out)
        elif ((a==2) and (k==14)):
            out = self.bn23(out)
        elif ((a==3) and (k==14)):
            out = self.bn24(out)
        elif ((a==3) and (k==16)):
            out = self.bn25(out)
        elif ((a==3) and (k==18)):
            out = self.bn26(out)            
        elif ((a==3) and (k==20)):
            out = self.bn27(out)
            
        if self.downsample is not None:
            identity = self.downsample(x, a, k)

        out += identity
        out = self.relu(out)

        return out, a, k


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bit, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, evaluate=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width, bit)
        self.conv2 = conv3x3(width, width, bit, stride, groups, dilation)
        self.conv3 = conv1x1(width, planes * self.expansion, bit=bit)

        # initialize the BN layer for each of the term budget
        self.bn10 = norm_layer(width)
        self.bn11 = norm_layer(width)
        self.bn12 = norm_layer(width)
        self.bn13 = norm_layer(width)
        self.bn14 = norm_layer(width)
        self.bn15 = norm_layer(width)
        self.bn16 = norm_layer(width)
        self.bn17 = norm_layer(width)
        
        self.bn20 = norm_layer(width)
        self.bn21 = norm_layer(width)
        self.bn22 = norm_layer(width)
        self.bn23 = norm_layer(width)
        self.bn24 = norm_layer(width)
        self.bn25 = norm_layer(width)
        self.bn26 = norm_layer(width)
        self.bn27 = norm_layer(width)
        
        self.bn30 = norm_layer(planes * self.expansion)
        self.bn31 = norm_layer(planes * self.expansion)
        self.bn32 = norm_layer(planes * self.expansion)
        self.bn33 = norm_layer(planes * self.expansion)
        self.bn34 = norm_layer(planes * self.expansion)
        self.bn35 = norm_layer(planes * self.expansion)
        self.bn36 = norm_layer(planes * self.expansion)
        self.bn37 = norm_layer(planes * self.expansion)
        ######################################################
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, a, k):
        identity = x
        out = self.conv1(x, a, k)

        if ((a==2) and (k==8)):
            out = self.bn10(out)
        elif ((a==2) and (k==10)):
            out = self.bn11(out)
        elif ((a==2) and (k==12)):
            out = self.bn12(out)
        elif ((a==2) and (k==14)):
            out = self.bn13(out)
        elif ((a==3) and (k==14)):
            out = self.bn14(out)
        elif ((a==3) and (k==16)):
            out = self.bn15(out)
        elif ((a==3) and (k==18)):
            out = self.bn16(out)
        elif ((a==3) and (k==20)):
            out = self.bn17(out)
            
        out = self.relu(out)
        out = self.conv2(out, a, k)

        if ((a==2) and (k==8)):
            out = self.bn20(out)
        elif ((a==2) and (k==10)):
            out = self.bn21(out)
        elif ((a==2) and (k==12)):
            out = self.bn22(out)
        elif ((a==2) and (k==14)):
            out = self.bn23(out)
        elif ((a==3) and (k==14)):
            out = self.bn24(out)
        elif ((a==3) and (k==16)):
            out = self.bn25(out)
        elif ((a==3) and (k==18)):
            out = self.bn26(out)
        elif ((a==3) and (k==20)):
            out = self.bn27(out)
            
        out = self.relu(out)
        out = self.conv3(out, a, k)

        if ((a==2) and (k==8)):
            out = self.bn30(out)
        elif ((a==2) and (k==10)):
            out = self.bn31(out)
        elif ((a==2) and (k==12)):
            out = self.bn32(out)
        elif ((a==2) and (k==14)):
            out = self.bn33(out)
        elif ((a==3) and (k==14)):
            out = self.bn34(out)
        elif ((a==3) and (k==16)):
            out = self.bn35(out)
        elif ((a==3) and (k==18)):
            out = self.bn36(out)
        elif ((a==3) and (k==20)):
            out = self.bn37(out)
            
        if self.downsample is not None:
            identity = self.downsample(x, a, k)

        out += identity
        out = self.relu(out)

        return out, a, k



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, bit=-1, evaluate=False):
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
        self.bn14 = norm_layer(self.inplanes)
        self.bn15 = norm_layer(self.inplanes)
        self.bn16 = norm_layer(self.inplanes)
        self.bn17 = norm_layer(self.inplanes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bit=bit, evaluate=evaluate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], bit=bit, evaluate=evaluate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], bit=bit, evaluate=evaluate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], bit=bit, evaluate=evaluate)
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, bit=-1, evaluate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            layer_subsample = [conv1x1(self.inplanes, planes * block.expansion, bit, stride),
                norm_layer(planes * block.expansion)]
            downsample = mySequential(*layer_subsample)
            
        layers = []
        layers.append(block(self.inplanes, planes, bit, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, evaluate=evaluate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, bit, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, evaluate=evaluate))

        return mySequential(*layers)

    def forward(self, x, a, k):

        x = self.conv1(x)
        if ((a==2) and (k==8)):
            x = self.bn10(x)
        elif ((a==2) and (k==10)):
            x = self.bn11(x)
        elif ((a==2) and (k==12)):
            x = self.bn12(x)
        elif ((a==2) and (k==14)):
            x = self.bn13(x)
        elif ((a==3) and (k==14)):
            x = self.bn14(x)
        elif ((a==3) and (k==16)):
            x = self.bn15(x)
        elif ((a==3) and (k==18)):
            x = self.bn16(x)            
        elif ((a==3) and (k==20)):
            x = self.bn17(x)
            
        x = self.relu(x)
        x = self.maxpool(x)

        x, a, k = self.layer1(x, a, k)
        x, a, k = self.layer2(x, a, k)
        x, a, k = self.layer3(x, a, k)
        x, a, k = self.layer4(x, a, k)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()


def _resnet(arch, block, layers, pretrained, progress, bit, evaluate, **kwargs):
    model = ResNet(block, layers, bit=bit, evaluate=evaluate, **kwargs)
    if pretrained:
        state_dict = load_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, bit=-1, evaluate=False, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, bit, evaluate, **kwargs)


def resnet50(pretrained=False, progress=True, bit=-1, evaluate=False, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, bit, evaluate,
                   **kwargs)
