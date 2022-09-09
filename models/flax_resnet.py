import torch
import jax
import jax.numpy as np
import flax.linen as nn
import jax.numpy as jnp


def conv7x7(out_planes, stride = 1):
    return nn.Conv(out_planes, kernel_size = (7, 7), strides = stride, use_bias = False)

def conv3x3(out_planes, stride = 1):
    return nn.Conv(out_planes, kernel_size = (3, 3), strides = stride, use_bias = False)

def conv1x1(out_planes, stride = 1):
    return nn.Conv(out_planes, kernel_size = (1, 1), strides = stride, use_bias = False)

class BasicBlock(nn.Module):
    planes: int
    stride: int = 1
    first_of_layer: bool = False
    norm_layer: nn.Module = nn.BatchNorm
    
    
    def setup(self):
        self.conv1 = conv3x3(self.planes, stride = self.stride)
        self.bn1 = self.norm_layer(use_running_average = False)
        self.conv2 = conv3x3(self.planes)
        self.bn2 = self.norm_layer(use_running_average = False)
        
        if self.stride != 1 or self.first_of_layer:
            self.downsample = True
            self.downsample_conv = conv1x1(self.planes, stride = self.stride)
            self.downsample_bn = self.norm_layer(use_running_average = False)
            
        else: self.downsample = None
            
    def __call__(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        out += identity
        out = nn.relu(out)
        
        return out
        
class Bottleneck(nn.Module):
    planes: int
    stride: int = 1
    first_of_layer: bool = False
    norm_layer: nn.Module = nn.BatchNorm
    expansion: int = 4
    
    
    def setup(self):
        self.conv1 = conv1x1(self.planes)
        self.bn1 = self.norm_layer(use_running_average = False)
        self.conv2 = conv3x3(self.planes, stride = self.stride)
        self.bn2 = self.norm_layer(use_running_average = False)
        self.conv3 = conv1x1(self.planes * self.expansion)
        self.bn3 = self.norm_layer(use_running_average = False)
        
        if self.stride != 1 or self.first_of_layer:
            self.downsample = True
            self.downsample_conv = conv1x1(self.planes * self.expansion, stride = self.stride)
            self.downsample_bn = self.norm_layer(use_running_average = False)
        else: self.downsample = None
            
            
        
    def __call__(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)
            
        out = out + identity
        out = nn.relu(out)
        
        return out
    
    
class ResNet(nn.Module):
    block: nn.Module
    layers: list
    num_classes: int = 100
    norm_layer: nn.Module = nn.BatchNorm
    block_type: str = 'Bottleneck'
        
    def setup(self):

        self.conv1 = nn.Conv(64, kernel_size = (7, 7), strides = 2, use_bias = False, padding = ((3, 3), (3, 3)))
        self.bn1 = self.norm_layer(use_running_average = False)
        
        self.layer1 = self.make_layer(self.block, 64,  self.layers[0], stride = 1)
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride = 2)
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride = 2)
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride = 2)
        
        self.fc = nn.Dense(self.num_classes)
        
        
    def make_layer(self, block, planes, blocks, stride = 1):
        assert stride in [1, 2]
        
        layers = []
        layers.append(block(planes, stride, first_of_layer = True))
        
        for i in range(1, blocks):
            layers.append(block(planes))
            
        return nn.Sequential(layers)
            
    def __call__(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        out = nn.max_pool(out, (3, 3), strides = (2, 2), padding = ((1, 1), (1, 1)))
        
        out = self.layer1(x)
        out = self.layer2(x)
        out = self.layer3(x)
        out = self.layer4(x)
        
        out= jnp.mean(out, axis=(1, 2))
        out = self.fc(out)
        
        return out
        

        
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])           
def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
    
        