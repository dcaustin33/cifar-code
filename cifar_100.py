import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class transformation(nn.Module):
    
    def __init__(self, crop_size: int = 32,
                       min_scale: int = .18,
                       max_scale: int = 1,
                       gaussian_prob: int = .2,
                       gray_scale_prob: int = .2,
                       horizontal_flip_prob: int = .5,
                       brightness: int = .4,
                       contrast: int = .4,
                       saturation: int = .2,
                       hue: int = .1,
                       color_jitter_prob: int = .4,
                       max_diff: int = 5,):
    
        
        super(transformation, self).__init__()
        self.resize_crop = transforms.RandomResizedCrop((crop_size, crop_size), scale=(min_scale, max_scale),
                                    interpolation=transforms.InterpolationMode.BICUBIC)
        
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        self.transform = transforms.Compose(
                            [transforms.RandomApply(
                                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                                    p=color_jitter_prob,
                                ),
                                transforms.RandomApply(
                                    [transforms.GaussianBlur(1)],
                                    p=gaussian_prob,
                                ),
                                transforms.RandomGrayscale(p=gray_scale_prob),
                                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                                transforms.Normalize(mean, std),
                            ]
                        )
        def forward(self, x):
            x = self.resize_crop(x)
            x = torch.clamp(x, 0, 1)
            return self.transform(x)

    
class CIFAR_100(Dataset):
    def __init__(self, train = True):
        if train:
            self.data =  torchvision.datasets.CIFAR100(root='../CIFAR-100', train=True,
                                        download=True)
        else:
            self.data = torchvision.datasets.CIFAR100(root='../CIFAR-100', train=False,
                                        download=True)
            
        self.classes = 100
            
    def __len__(self):
        return self.data.__len__()
        
    def __getitem__(self, idx):
        image, label = self.data.__getitem__(idx)
        image = tfs.ToTensor()(image)
        return image, label
    
class CIFAR_100_transformations(Dataset):
    def __init__(self, train = True, **kwargs):
        if train:
            self.data =  torchvision.datasets.CIFAR100(root='../CIFAR-100', train=True,
                                        download=True)
        else:
            self.data = torchvision.datasets.CIFAR100(root='../CIFAR-100', train=False,
                                        download=True)
            
        self.classes = 100
        self.transform = transformation(**kwargs)
            
    def __len__(self):
        return self.data.__len__()
        
    def __getitem__(self, idx):
        image, label = self.data.__getitem__(idx)
        data = {}
        image = tfs.ToTensor()(image)
        data['image1'] = self.transform(image)
        data['image2'] = self.transform(image)
        data['label'] = label
        return data