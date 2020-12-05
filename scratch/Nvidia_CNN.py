import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=24,kernel_size=5,stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=24,out_channels=36,kernel_size=5,stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=36,out_channels=48,kernel_size=5,stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=48,out_channels=64,kernel_size=3),
            nn.ELU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3),
            nn.Dropout(0.25)
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=1280, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )

    
    def forward(self,inp):
        inp=self.conv(inp)
        inp=inp.view(-1,1280)
        inp=self.linear(inp)    
        return inp.squeeze(1)





        
        
        
    
