import torchvision
import torch.nn as nn
import torch
import math

def ResNet18(cifar = True):
    model = torchvision.models.resnet18()
    model.fc = nn.Identity()
    if cifar:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.maxpool = nn.Identity()
    return model