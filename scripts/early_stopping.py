import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
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
        elif 1- (score/self.best_score) > delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} (Current loss: {val_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'New best model:({self.val_loss_min:.6f} --> {val_loss:.6f}), Impr: {1- (score/self.best_score)*100}%')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss