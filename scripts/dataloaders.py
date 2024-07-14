
import pandas as pd
import numpy as np
import os
import torch 

from .reax_ff_data import bo
from scripts.making_df import *
from scripts.matrix_function import *
from scripts.params import *
from scripts.utils import making_rgb_numerically, creating_images
from torch.utils.data import DataLoader, TensorDataset


def read_files():
    '''Read the file, if file does not exist then exctract information from .tar file'''
    try:
        files = pd.read_csv(f'{PATH}/{DB}.csv')
    except FileNotFoundError:
        making_df()
        files=pd.read_csv(f'{PATH}/{DB}.csv')

        try:
            os.system('mkdir data')
            os.system(f'tar -xvf dsgdb9nsd.xyz.tar.bz2 -C {PATH}/data')
            making_df()
            files = pd.read_csv(f'{PATH}/{DB}.csv')         
        except:
            print('Unknown error occured during loading the databse file' )
    return files

def dataloader_ffn():
    raw_data = read_files()

    n=len(raw_data) -1
    def creating_images(start, end, bo, ds, step=1):
        l = np.empty((0, 32*32*3 + 1))
        for chem in range(start, end+1, step):
            print(f'\rCreating dataset:{chem/n*100:.2f}%',end='')
            flatten = making_rgb_numerically(chem, bo, ds, scaling=False)
            fl = np.hstack(flatten).reshape(-1, 1)
            fl = np.append(fl, ds[PREDICTED_VALUE].iloc[chem])
            l = np.append(l, [fl], axis=0)
        return l

    def create_dataloader(X, y, batch_size=BATCH_SIZE):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    matrix = creating_images(0, n, bo, raw_data)
    training_volume = int(len(matrix) * TRAIN_TEST_SPLIT)

    X_train, y_train = matrix[:training_volume, :-1], matrix[:training_volume, -1]
    X_test, y_test = matrix[training_volume:, :-1], matrix[training_volume:, -1]

    train_loader = create_dataloader(X_train, y_train)
    val_loader = create_dataloader(X_test, y_test)

    return train_loader, val_loader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        name = f'{DB}.csv'
        self.df = pd.read_csv(os.path.join(PATH, name))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = f"{self.df['ID'].iloc[idx]}.png"
        try: 
            image = Image.open(f'{TRAIN_DIR_NAME}/{img_name}').convert('RGB') 

        except FileNotFoundError:
            image = Image.open(f'{TEST_DIR_NAME}/{img_name}').convert('RGB')

        label = self.df[PREDICTED_VALUE].iloc[idx]
        label = torch.tensor(label, dtype=torch.float32)#.to(DEVICE)

        if self.transform:
            image = self.transform(image)

        return image, label
     
def dataloader_conv(n = 0):
    raw_data = read_files()
    if n ==0: n=len(raw_data) -1
    if DELETE == True: 
        creating_images(0, n, bo, raw_data) 
    else: 
        print(f'Program did not create new images because DELETE parameter is set to False')

    train_transforms = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    train_dataset = CustomDataset(transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CustomDataset(transform=train_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


    return train_loader, val_loader


