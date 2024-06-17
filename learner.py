from scripts.params import *

import torch
import torch.nn as nn
import torch.optim as optim
import os

from scripts.dataloaders import *
from models.conv import *
from models.flatten import *

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
            losses.append(epoch_loss)

            with torch.no_grad():
                original_value= torch.cat([x[1] for x in dl[1]]).to(DEVICE) #in the form of one dimensional tensor
                predicted_value = torch.cat([model(x[0].to(DEVICE)) for x in dl[1]]).t()

                acc = sum(sum(torch.abs(original_value-predicted_value)))/predicted_value.size()[1]*27211
            
            accuracies.append(acc.item())
            print(f'Epoch [{e+1}/{EPOCHS}], Loss: {epoch_loss:.6f}, Acc: {acc:.2f} meV', file = file)

            del inputs, targets, outputs, loss, original_value, predicted_value
            torch.cuda.empty_cache()

        print(f"Log file has been saved as: {PATH}/{LOG_FILE}")
        torch.save(model.state_dict(), f"{PATH}/{LOG_FILE.replace('log','pth')}")

        
        print('--------', file=file)
        print(f"Model has been saved as {LOG_FILE.replace('log','pth')}", file=file)
        print('--------', file=file)
        print(f"Losses values:{losses}", file=file)
        print(f"Accuracy values:{accuracies}", file=file)
        from scripts.params import __all__
        print('--------', file=file)
        print(f'Copy of a params.py settings:', file=file)
        for name in __all__:
            print(f"{name} = {globals()[name]}", file=file)

    return losses, accuracies
