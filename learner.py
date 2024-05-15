from scripts.params import *

import torch
import torch.nn as nn
import torch.optim as optim

from scripts.dataloaders import *
from models.conv import *
from models.flatten import *

def learner(dl, model):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    losses = []
    accuracies = []

    for e in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in dl[0]:
            optimizer.zero_grad()
            outputs = model(inputs).to(torch.float32)
            loss = criterion(outputs.t().view(-1), targets.to(torch.float32))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dl[0].dataset)

        original_value= torch.cat([x[1] for x in dl[1]]) #in the form of one dimensional tensor
        predicted_value = torch.cat([model(x[0]) for x in dl[1]]).t()

        acc = sum(sum(torch.abs(original_value-predicted_value)))/predicted_value.size()[1]*27211

        print(f'Epoch [{e+1}/{EPOCHS}], Loss: {epoch_loss:.6f}, Acc: {acc:.2f} meV')
        losses.append(epoch_loss)
        accuracies.append(acc)
    
    return losses, accuracies
