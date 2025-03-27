
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
from torch.utils.data import DataLoader, TensorDataset, random_split


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
    def __init__(self, file_list = None, data_dir = TRAIN_DIR_NAME, transform=None):
        self.transform = transform
        name = f'{DB}.csv'
        self.df = pd.read_csv(os.path.join(PATH, name)) 
        self.files = os.listdir(os.path.join(PATH, data_dir)) if file_list is None else file_list
        self.labels = {row['ID']: row[PREDICTED_VALUE] for _, row in self.df.iterrows()}
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.files)#, len(self.df)

    def __getitem__(self, idx):
        #img_name = f"{self.df['ID'].iloc[idx]}.png"
        img_name = f"{self.files[idx]}"
        image = Image.open(f'{PATH}/{self.data_dir}/{img_name}').convert('RGB') 

        img_id = img_name.split('.')[0]  
        #print(len(self.files), idx, img_id)

        label = self.labels.get(img_id, -1)
        label = torch.tensor(label, dtype=torch.float32)

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

    trans = transforms.Compose([transforms.Resize((RESIZE,RESIZE)), transforms.ToTensor()])

    all_files = os.listdir(f"{PATH}/{TRAIN_DIR_NAME}")
    train_size = int(len(all_files) * TRAIN_VAL_SPLIT)
    val_size = len(all_files) - train_size

    #train_files, val_files = train_test_split(os.listdir(f"{PATH}/{TRAIN_DIR_NAME}"), test_size=1-TRAIN_VAL_SPLIT, random_state=42)
    train_files, val_files = random_split(all_files, [train_size, val_size])

    #create datasets (based on TRAIN_DIR_NAME)
    train_dataset = CustomDataset(transform=trans, data_dir=TRAIN_DIR_NAME, file_list=train_files)
    val_dataset = CustomDataset(transform=trans, data_dir=TRAIN_DIR_NAME, file_list=val_files)

    #create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #test dataset (based on TEST_DIR_NAME)
    test_dataset = CustomDataset(transform=trans, data_dir=TEST_DIR_NAME)
    test_loader = DataLoader(test_dataset, batch_size=int(len(os.listdir(os.path.join(PATH, TEST_DIR_NAME)))/CYCLE), shuffle=False)

    return train_loader, val_loader, test_loader, raw_data


