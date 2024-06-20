import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import torch
import pandas as pd
import numpy as np

from scripts.params import *
from models.flatten import SimpleNN
from models.conv import SCNN

from scripts.dataloaders import dataloader_conv, dataloader_ffn
from learner import learner


dl = dataloader_conv()
#model = SCNN()

from torchvision.models  import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT).to(DEVICE)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

#model = SCNN()
m = learner(dl, model)
