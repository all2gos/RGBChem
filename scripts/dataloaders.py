
import pandas as pd
import numpy as np
import os, shutil
import torch 
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

from .reax_ff_data import bo
from scripts.making_df import *
from scripts.matrix_function import *
from scripts.params import *
from scripts.utils import making_rgb_numerically, creating_images
from torch.utils.data import DataLoader, TensorDataset


def read_files():
    '''Read the file, if file does not exist then exctract information from .tar file'''

    print('reading files')
    db_file_exist = os.path.exists(f'{PATH}/{DB}.csv')
    data_dir_exist = os.path.exists(f"{PATH}/data")

    if data_dir_exist:
        print(os.getcwd())
        shutil.rmtree(f"{PATH}/data") 

    if not db_file_exist:

        os.mkdir(f"{PATH}/data")
        images_exists = len(os.listdir(f"{PATH}/data"))>10
        if not images_exists:
            os.system(f'tar -xvf dsgdb9nsd.xyz.tar.bz2 -C {PATH}/data')
                
        making_df()
        
    files = pd.read_csv(f'{PATH}/{DB}.csv')

    return files
  
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        name = f'{DB}.csv'
        self.df = pd.read_csv(os.path.join(PATH, name))
        self.files = os.listdir(os.path.join(PATH, TRAIN_DIR_NAME)) + os.listdir(os.path.join(PATH, TEST_DIR_NAME))
        self.labels = {row['ID']: row[PREDICTED_VALUE] for _, row in self.df.iterrows()}
        
    def __len__(self):
        return len(self.files)#, len(self.df)

    def __getitem__(self, idx):
        #img_name = f"{self.df['ID'].iloc[idx]}.png"
        img_name = f"{self.files[idx]}"
        try: 
            image = Image.open(f'{PATH}/{TRAIN_DIR_NAME}/{img_name}').convert('RGB') 

        except FileNotFoundError:
            image = Image.open(f'{PATH}/{TEST_DIR_NAME}/{img_name}').convert('RGB')

        label = self.df[PREDICTED_VALUE].iloc[idx]
        label = torch.tensor(label, dtype=torch.float32)#.to(DEVICE)

        img_id = img_name.split('.')[0]  
        print(len(self.files), idx, img_id)

        label = self.labels.get(img_id, -1)
        label = torch.tensor(label, dtype=torch.float32)#.to(DEVICE)

        
        if self.transform:
            image = self.transform(image)

        return image, label
     
def dataloader_conv(n = 0):
    raw_data = read_files()
    if n ==0: n=len(raw_data) -1
    if DELETE == True: 
        creating_images(0, n, bo, raw_data, STEP) 
    else: 
        print(f'Program did not create new images because DELETE parameter is set to False')

    train_transforms = transforms.Compose([transforms.Resize((RESIZE,RESIZE)), transforms.ToTensor()])
    train_dataset = CustomDataset(transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CustomDataset(transform=train_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


    return train_loader, val_loader, raw_data


