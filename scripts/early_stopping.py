import torch
import numpy as np

from scripts.params import *
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.01, path='checkpoint.pt'):
        """
            patience (int): how long wait to the next improvment
            delta (float): minimal change (fraction)
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.val_loss_min * (1 - self.delta):
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} (Current loss: {val_loss:.4f} Impr: {(1-val_loss/self.val_loss_min)*100:.2f}%)')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Imprv: {(100*(1-(val_loss/self.val_loss_min))):.2f}% Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

