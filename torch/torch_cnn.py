import torchvision
import torch.nn as nn
import torch
import math
import sys
sys.path.append("../") # go to parent dir
import ResNet18 as ResNet18


class res18(nn.Module):
    
    def __init__(self, 
                 input_size: int = (3, 32, 32), 
                 classes: int = 100):
        
        super().__init__()
        
        self.network = ResNet18.ResNet18()
            
        example = torch.ones(input_size)
        self.feature_dim = (self.network(example.unsqueeze(0))).shape[1]

        self.network.fc = nn.Linear(self.feature_dim, classes)
            
            
            
    def forward(self, x, second_video = False, seconds = None, arrow_time = None, diff = None):
        
        a1 = self.network(x)
        #a2 = self.classifier(a1)
        return a1