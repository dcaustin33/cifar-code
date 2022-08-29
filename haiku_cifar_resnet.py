from cmath import exp
from operator import is_
import torch
import jax
import jax.numpy as np
import haiku as hk
import jax.numpy as jnp


def conv7x7(out_planes, stride = 1):
    return hk.Conv2D(out_planes, kernel_shape = 7, stride = stride, with_bias = False)

def conv3x3(out_planes, stride = 1):
    return hk.Conv2D(out_planes, kernel_shape = 3, stride = stride, with_bias = False)

def conv1x1(out_planes, stride = 1):
    return hk.Conv2D(out_planes, kernel_shape = 1, stride = stride, with_bias = False)

class BasicBlock(hk.Module):
    
    def __init__(self,
                 planes: int,
                 stride: int = 1,
                 first_of_layer: bool = False):
        super().__init__(name="BasicBlock")
        self.planes = planes
        self.stride = stride
        self.first_of_layer = first_of_layer


        self.conv1 = conv3x3(self.planes, stride = self.stride)
        self.bn1 = hk.BatchNorm(create_offset=True, 
                              create_scale=True, 
                              decay_rate = .9)
        self.conv2 = conv3x3(self.planes)
        self.bn2 = hk.BatchNorm(create_offset=True, 
                              create_scale=True, 
                              decay_rate = .9)
        
        if self.stride != 1 or self.first_of_layer:
            self.downsample = True
            self.downsample_conv = conv1x1(self.planes, stride = self.stride)
            self.downsample_bn = hk.BatchNorm(create_offset=True, 
                              create_scale=True, 
                              decay_rate = .9)
            
        else: self.downsample = None
            
    def __call__(self, x, is_training):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, is_training=is_training)
        out = jax.nn.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out, is_training=is_training)
        
        if self.downsample:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity, is_training=is_training)

        out += identity
        out = jax.nn.relu(out)
        
        return out
        
class Bottleneck(hk.Module):
    
    
    def __init__(self,
                 planes: int,
                 stride: int = 1,
                 first_of_layer: bool = False,
                 expansion: int = 4):
        super().__init__(name="BottleneckBlock")
        self.planes = planes
        self.stride = stride
        self.first_of_layer = first_of_layer
        self.expansion = expansion


        self.conv1 = conv1x1(self.planes)
        self.bn1 = hk.BatchNorm(create_offset=True, 
                              create_scale=True, 
                              decay_rate = .9)
        self.conv2 = conv3x3(self.planes, stride = self.stride)
        self.bn2 = hk.BatchNorm(create_offset=True, 
                              create_scale=True, 
                              decay_rate = .9)
        self.conv3 = conv1x1(self.planes * self.expansion)
        self.bn3 = hk.BatchNorm(create_offset=True, 
                              create_scale=True, 
                              decay_rate = .9)
        
        if self.stride != 1 or self.first_of_layer:
            self.downsample = True
            self.downsample_conv = conv1x1(self.planes * self.expansion, stride = self.stride)
            self.downsample_bn = hk.BatchNorm(create_offset=True, 
                              create_scale=True, 
                              decay_rate = .9)
        else: self.downsample = None
            
            
        
    def __call__(self, x, is_training):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out, is_training=is_training)
        out = jax.nn.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out, is_training=is_training)
        out = jax.nn.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out, is_training=is_training)
        
        if self.downsample:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity, is_training=is_training)
            
        out = out + identity
        out = jax.nn.relu(out)
        
        return out
    
class block_group(hk.Module):

    def __init__(self, block, planes, blocks, stride = 1):
        super().__init__()
        assert stride in [1, 2]
        self.layers = []
        if planes == 64:
            self.layers.append(block(planes, stride, first_of_layer = False))
        else:
            self.layers.append(block(planes, stride, first_of_layer = True))
        
        for i in range(1, blocks):
            self.layers.append(block(planes))
            
    def __call__(self, x, is_training):
        out = x
        for layer in self.layers:
            out = layer(out, is_training=is_training)
        return out

class ResNet(hk.Module):
        
    def __init__(self, 
                 block: hk.Module, 
                 layers: list,
                 num_classes: int = 100):
        super().__init__(name="ResNet")
        self.block = block
        self.layers = layers
        self.num_classes = num_classes

        self.conv1 = hk.Conv2D(64, kernel_shape = 3, stride = 1, with_bias = False, padding = ((2, 2), (2, 2)))
        self.bn1 = hk.BatchNorm(create_offset=True, 
                              create_scale=True, 
                              decay_rate = .9)
        
        self.blocks = []

        planes = 64
        for i, num_layers in enumerate(self.layers):
            stride = 1 if i == 0 else 2
            self.blocks.append(block_group(self.block, planes, num_layers, stride))
            planes = planes * 2
        '''self.layer1 = self.make_layer(self.block, 64,  self.layers[0], stride = 1)
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride = 2)
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride = 2)
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride = 2)'''
        
        self.fc = hk.Linear(self.num_classes)
        
        
    def make_layer(self, block, planes, blocks, stride = 1):
        assert stride in [1, 2]
        
        layers = []
        if planes == 64:
            layers.append(block(planes, stride, first_of_layer = False))
        else:
            layers.append(block(planes, stride, first_of_layer = True))

        #layers.append(block(planes, stride, first_of_layer = True))
        
        for i in range(1, blocks):
            layers.append(block(planes))
            
        return layers
            
    def __call__(self, x, is_training = True):
        out = self.conv1(x)
        out = self.bn1(out, is_training=is_training)
        out = jax.nn.relu(out)

        #no max pooling at this resolution
        #out = nn.max_pool(out, (3, 3), strides = (2, 2), padding = ((1, 1), (1, 1)))
        for block in self.blocks:
            out = block(out, is_training = is_training)
        '''for layer in self.layer1: 
            out = layer(out, is_training=is_training)
        for layer2 in self.layer2: 
            out = layer2(out, is_training=is_training)
        for layer in self.layer3: out = layer(out, is_training=is_training)
        for layer in self.layer4: out = layer(out, is_training=is_training)'''
        
        out= jnp.mean(out, axis=(1, 2))
        
        out = self.fc(out)
        
        return out
        

        
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2]) 
def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
    
        