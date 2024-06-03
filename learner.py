from scripts.params import *

import torch
import torch.nn as nn
import torch.optim as optim
import os

from scripts.dataloaders import *
from models.conv import *
from models.flatten import *

def initial_info():
    '''
    A function that prints a lot of information and writes it to the log
    '''

    print('Welcome to the file containing information about training the model', file=LOG_FILE)

    print(f'All of ')
def learner(dl, model):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    losses = []
    accuracies = []
    model = model.to(DEVICE)
    with open(LOG_FILE, 'w') as file:        
        for e in range(EPOCHS):
            print(f'\rModel is training: {e+1}/{EPOCHS}', end='')
            model.train()
            running_loss = 0.0

            for inputs, targets in dl[0]:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.t().view(-1), targets.to(torch.float32))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dl[0].dataset)

            original_value= torch.cat([x[1] for x in dl[1]]) #in the form of one dimensional tensor
            predicted_value = torch.cat([model(x[0]) for x in dl[1]]).t()

            acc = sum(sum(torch.abs(original_value-predicted_value)))/predicted_value.size()[1]*27211
            
            losses.append(epoch_loss)
            accuracies.append(acc)
            print(f'Epoch [{e+1}/{EPOCHS}], Loss: {epoch_loss:.6f}, Acc: {acc:.2f} meV', file = file)

        print(f"{PATH}/{LOG_FILE.replace('log','pth')}")
        torch.save(model.state_dict(), f"{PATH}/{LOG_FILE.replace('log','pth')}")

        
        print(f"Model has been saved as {LOG_FILE.replace('log','pth')}", file=file)

        print(f"Losses values:{losses}", file=file)
        print(f"Accuracy values:{accuracies}", file=file)
        print(f'Copy of a params.py settings:', file=file)
        with open(f'{PATH}/scripts/params.py', 'r') as f:
            l = f.readlines()
        for line in l:
            print(line, file=file)

    return losses, accuracies
