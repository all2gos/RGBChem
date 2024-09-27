import pandas as pd
from scripts.dataloaders import dataloader_conv
from scripts.params import *
from fastai.vision.all import *
from scripts.params import *
import torch
import time
import subprocess

#this function perform database creation, .png files creation and DataLoader and Dataset PyTorch object creation. Moreover it is possible to create
#a fastai workflow build on that components which we will show you in this demo.

dl = dataloader_conv()
ds = pd.read_csv(f'{PATH}/{DB}.csv')

def get_list(path):
    l = []
    for plik in os.listdir(path):
        if os.path.isfile(os.path.join(path, plik)):
            l.append(plik)
    return l

def get_x(r): return f"{path}/{r['ID']}.png"
def get_y(r): return float(r[PREDICTED_VALUE])

path = f'{PATH}/{TRAIN_DIR_NAME}'
get_image_files = get_list(path)

f = os.listdir(f"{PATH}/{TRAIN_DIR_NAME}")
f = [x[:-4] for x in f]

filtered = ds[ds.ID.isin(f)]
print(f'{len(filtered)} out of {len(ds)} samples were selected')

dblock = DataBlock(blocks=(ImageBlock, RegressionBlock),
                   get_x=get_x, get_y=get_y,
                   splitter=RandomSplitter(valid_pct=0.1, seed=42),
                   item_tfms = Resize(RESIZE)).dataloaders(filtered, bs=BATCH_SIZE)

learn = vision_learner(dblock, resnet18, metrics=mae, lr=LEARNING_RATE)
saving_callbacks = SaveModelCallback(monitor='valid_loss', comp=np.less, min_delta=DELTA, fname=f"{PATH}/{LOG_FILE.replace('.log','checkpoint_fastai')}")
early_stopping_cb = EarlyStoppingCallback(monitor='valid_loss', comp=np.less, min_delta=DELTA, patience=PATIENCE)

class WaitTimeCallback(Callback):
    def get_battery_info(self):
        result = subprocess.check_output(['acpi', '-b'], text=True)
        try:
            idx = result.index('%') 
            battery_level = int(result[idx-3:idx])
            total_seconds = (80-battery_level)*40 if battery_level < 98 else 40
        except ValueError:
            print(f"Info about charge level not found")
            total_seconds = 250

        return battery_level, max(0,total_seconds)

    def after_epoch(self):
        info = self.get_battery_info()
        print(f'Dealing with Battery level: charge level: {info[0]}. Waiting for {info[1]} seconds...')
        time.sleep(info[1])

learn.fine_tune(EPOCHS, cbs=[early_stopping_cb, saving_callbacks, WaitTimeCallback()]) 
learn.export(f"{PATH}/{LOG_FILE.replace('.log','_fastai.pkl')}")

print('Validation...')

test_files = os.listdir(f"{PATH}/{TEST_DIR_NAME}")
test_data = [PILImage.create(f'{PATH}/{TEST_DIR_NAME}/{file}') for file in test_files]
test_dl = learn.dls.test_dl(test_data)
preds, _ = learn.get_preds(dl=test_dl)

err = []


for idx in range(1,len(test_files),max(CYCLE-1,1)):
    print(f'\r{idx/len(test_files):.2f}',end='')
    actual = ds[PREDICTED_VALUE].loc[ds.ID == test_files[idx][:-4]].values[0]
    err.append(np.abs(preds[idx].item() - float(actual)))

print(f"\n Average prediction error on test set: {sum(err)/len(err)*27211:.2f} meV")

with open(LOG_FILE, 'w+') as file:

    print(f"\n Average prediction error on test set: {sum(err)/len(err)*27211:.2f} meV", file=file)    
    print(f'Copy of a params.py settings:', file=file)
    from scripts.params import __all__
    for name in __all__:
        print(f"{name} = {globals()[name]}", file=file)

