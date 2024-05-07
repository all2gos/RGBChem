
import pandas as pd
import numpy as np
import os

from reax_ff_data import bo
from .matrix_function import *
from .params import *
from .utils import making_rgb_numerically
from torch.utils.data import DataLoader, TensorDataset



def dataloader_ffn():
    raw_data = pd.read_csv(f'{PATH}/{DB}.csv')

    def creating_images(start, end, bo, ds, split=0.1, step=1, verbose = False):
        l = np.empty((0, 32*32*3 + 1))
        for chem in range(start, end+1, step):
            flatten = making_rgb_numerically(chem, bo, ds, scaling=False)
            fl = np.vstack((flatten[0].reshape(32*32,1), flatten[1].reshape(32*32,1), flatten[2].reshape(32*32,1)))
            fl = np.append(fl, ds[PREDICTED_VALUE].iloc[chem])
            l = np.append(l, [fl], axis=0)
        return l

    matrix = creating_images(0, len(raw_data)-1, bo, raw_data,  0.1, 1)

    training_volume = int(matrix.shape[1]*TRAIN_TEST_SPLIT)

    X_train = matrix[:training_volume, :-1]
    y_train = matrix[:training_volume, -1]

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_test = matrix[training_volume:, :-1]
    y_test = matrix[training_volume:, -1]

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader