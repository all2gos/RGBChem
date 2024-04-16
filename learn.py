import os
import argparse
import numpy as np
from fastai.vision.all import *
import pandas as pd
import ssl
from params import *

ssl._create_default_https_context = ssl._create_unverified_context

def get_list(path):
    l = []
    for plik in os.listdir(path):
        if os.path.isfile(os.path.join(path, plik)):
            l.append(plik)
    return l

path = f'{PATH}/{TRAIN_DIR_NAME}'
get_image_files = get_list(path)

def get_x(r): return f"{path}/{r['ID']}.png"
def get_y(r): return float(r[PREDICTED_VALUE])

ds = pd.read_csv(f'{PATH}/{DB}.csv')
f = os.listdir(f"{PATH}/{TRAIN_DIR_NAME}")
f = [x[:-4] for x in f]

print(len(ds), SAMPLE)
filtered = ds[ds.ID.isin(f)]

if SAMPLE != 1:
    filtered = filtered.sample(n=len(filtered)*SAMPLE)
print(f'{len(filtered)} out of {len(ds)} samples were selected')

dblock = DataBlock(blocks=(ImageBlock, RegressionBlock),
                   get_x=get_x, get_y=get_y,
                   splitter=RandomSplitter(valid_pct=0.1, seed=42)).dataloaders(filtered)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--learn', dest='learn_file', default=None, help='Path to the file to load the model from')
parser.add_argument('--export', dest='export_file', default='name', help='File name for exporting the model (without extension)')
parser.add_argument('--arch', dest='architecture', default='resnet18', help='ResNet architecture to use')
parser.add_argument('--epochs', dest='num_epochs', type=int, default=24, help='Number of epochs for fine-tuning (default: 20)')

args = parser.parse_args()

learn = vision_learner(dblock, eval(args.architecture), metrics=mae)

if args.learn_file:
    learn.load(args.learn_file)


learn.fine_tune(args.num_epochs)
learn.export(args.export_file+'.pkl')

# Validation
err = []
test_files = os.listdir(f"{PATH}/{TEST_DIR_NAME}")

for idx in range(len(test_files)):
    num, _, probs = learn.predict(PILImage.create(f'{PATH}/{TEST_DIR_NAME}/{test_files[idx]}'))
    actual = ds[PREDICTED_VALUE].loc[ds.ID == test_files[idx][:-4]].values[0]
#    print(f"Prediction: {idx}/{len(test_files)}, model says: {num}, actually this is: {actual}: , The error is: {np.abs(num-float(actual))} ")
    err.append(np.abs(num-float(actual)))

print(f"Average error is {sum(err)/len(err)} meV")
learn.export(fname=args.export_file + '.pkl')

with open(args.export_file + '.log', 'w') as file:
    print(f'Architecture {args.export_file}: {args.architecture} fine_tuned for {args.num_epochs}', file=file)
    print(f"Average error on test set is {sum(err)/len(err)} meV", file=file)

