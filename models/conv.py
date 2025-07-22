import torch
import torch.nn as nn
from scripts.params import *

SIZE_OF_IMG = max(MATRIX_SIZE, RESIZE)

class S1CNN(nn.Module):
    def __init__(self):
        super(S1CNN,self).__init__()
        self.c1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.fc1 = nn.Linear(6*int(SIZE_OF_IMG/2)*int(SIZE_OF_IMG/2),8*16)
        self.fc2 = nn.Linear(8*16, 32)
        self.fc3 = nn.Linear(32, 1)


        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=.2)

    def forward(self, img):
        x = self.c1(img)
        x = self.relu(self.max_pool(x))
        x = torch.flatten(x,1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class S2CNN(nn.Module):
    def __init__(self):
        super(S2CNN, self).__init__()
        
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.c2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1, padding=2)
        
        self.fc1 = nn.Linear(12 * int(SIZE_OF_IMG / 4) * int(SIZE_OF_IMG / 4), 8 * 16)
        self.fc2 = nn.Linear(8 * 16, 32)
        self.fc3 = nn.Linear(32, 1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, img):
        x = self.c1(img)
        x = self.relu(self.max_pool(x))
        x = self.c2(x)
        x = self.relu(self.max_pool(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x