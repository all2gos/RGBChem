import pandas as pd
from scripts.dataloaders import dataloader_conv
from scripts.params import *
import torch
import torchvision.models as models
import random, os, time, subprocess
import torch.optim as optim
import torch.nn as nn
import numpy as np
from scripts.utils import welcome_message

from models.conv import *

import logging
from scripts.logging import setup_logging
setup_logging()

def move_one_image_to_the_outside():
    '''Function that after training procedure choose one .png and move it to the external file'''

    img_list= os.listdir(f"{PATH}/train")
    img_name = img_list[random.randint(1,len(img_list))]

    os.system(f"mv {PATH}/train/{img_name} {PATH}")
    print(f'File {img_name} have been successfully moved to {PATH}')

def get_list(path):
    l = []
    for plik in os.listdir(path):
        if os.path.isfile(os.path.join(path, plik)):
            l.append(plik)
    return l

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
    
#training step
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, targets, names in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs).squeeze()
        #print(outputs.size(), targets.size())
        loss = criterion(outputs,  targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)


#validation step
def validate(model, valid_loader, criterion, device, verbose=False):
    model.eval()
    running_loss = 0.0
    predictions, targets = [], []
    batch_cnt = 1
    loss_container = []
    with torch.no_grad():
        for inputs, targets, names in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            #print(inputs.size(), targets.size())
            #print(inputs)
            outputs = model(inputs).squeeze()
            #print(outputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            if verbose: 
                loss_container.append(round(loss.item(),5))
            batch_cnt +=1

        outputs, targets = outputs.cpu().numpy(), targets.cpu().numpy()
        if verbose:
            individual_loss = list(zip(names, loss_container))
            print(f'Individual losses: {individual_loss}')

    return running_loss / len(valid_loader), mae(np.array(targets), np.array(outputs)), np.var(np.array(outputs))/np.var(np.array(targets))

if __name__ == '__main__':

    welcome_message()

    #load data points
    train_loader, val_loader, test_loader, raw_data = dataloader_conv()
    ds = pd.read_csv(f'{PATH}/{DB}.csv')

    from scripts.params import __all__
    print(f"All params list")
    for name in __all__:
        print(f"{name} = {globals()[name]}")

    path = f'{PATH}/{TRAIN_DIR_NAME}'
    get_image_files = get_list(path)

    f = os.listdir(f"{PATH}/{TEST_DIR_NAME}")
    print(f'Number of images in {TEST_DIR_NAME} directory: {len(f)}')


    f = os.listdir(f"{PATH}/{TRAIN_DIR_NAME}")
    print(f'Number of images in {TRAIN_DIR_NAME} directory: {len(f)}')
    f = [x[:-4] for x in f]

    print(f[:4])
    print(ds.ID[:4])
    filtered = ds[ds.ID.isin(f)]

    compounds_above_size = len(filtered[filtered['n_atoms'] >= RESIZE])

    if compounds_above_size > 0:
        print(f"Warning! {compounds_above_size} out of {len(filtered)} compounds have more than {RESIZE} atoms. This may lead to problems with training procedure.") 

    print(f'{len(filtered)} out of {len(ds)} samples were selected for training procedure')

    not_selected = ds[~ds.ID.isin(f)]
    print(f"Compounds that have been not selected for training procedure:{not_selected.ID.values}")

    #load model
    if MODEL in ['S1CNN()','S2CNN()']:
        model = eval(MODEL)
    elif MODEL in ['vgg19_bn','VGG19_BN','VGG19','vgg19']:
        model = models.vgg19_bn(weights='IMAGENET1K_V1')
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1],  #changing last layer to one neuron
            nn.Linear(4096, 1)  
        )
    elif MODEL in ['VGG16','VGG16_BN','vgg16','vgg16_bn']:
        model = models.vgg16_bn(weights='IMAGENET1K_V1')
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1],  #changing last layer to one neuron
            nn.Linear(4096, 1)  
        )
    elif MODEL in ['resnet18', 'ResNet18', 'RESNET18']:
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        raise ValueError("Unknown model type")

    #gpu migration
    model.to(DEVICE)

    #optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    criterion = nn.L1Loss()


    best_valid_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = PATIENCE
    checkpoint_path = f"{PATH}/{LOG_FILE.replace('.log','checkpoint_pytorch.pth')}"

    #training loop
    cnt = 0
    for epoch in range(EPOCHS):
        cnt +=1
        t = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        valid_loss, valid_mae, valid_var= validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Validation MAE: {valid_mae:.4f} Pred/targets variance: {valid_var:.4f} Time: {int(time.time()-t)}s")
        
        #early stopping
        if valid_loss < best_valid_loss - DELTA:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)  #save new best model
            print(f"Model saved to {checkpoint_path}")
        else:
            if epoch > WAIT_UNTIL: #after 100 epochs we start counting patience
                patience_counter += 1
                print(f"{patience_counter}/{early_stopping_patience} epochs without improvement")
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

    #test 

    print(f'Final evaluation based on separate test set from {TEST_DIR_NAME} directory')
    print(f"Number of test samples: {len(test_loader.dataset)}")
    test_loss, test_mae, test_var= validate(model, test_loader, criterion, DEVICE, verbose=True)
    print(f"Test MAE: {test_mae:.4f} Pred/targets variance: {test_var:.4f}")
    logging.info(f"Test MAE: {test_mae:.4f} Pred/targets variance: {test_var:.4f}")


    from scripts.automated_training_and_db_handling import places
#   places(test_mae*1000, cnt)

