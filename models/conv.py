import torch
import torch.nn as nn

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN,self).__init__()
        self.c1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.fc1 = nn.Linear(6*16*16,8*16)
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
    

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x