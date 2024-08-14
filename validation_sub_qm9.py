import torch
from fastai.vision.all import *
from scripts.params import *


#loading
model_path = 'models/qm9Aresnet18_avg_e512_bandgap_bs64_shuffle_partial_1checkpoint_fastai.pth'

ds = pd.read_csv(f'{PATH}/{DB}.csv')[1000:]

def get_list(path):
    l = []
    for plik in os.listdir(path):
        if os.path.isfile(os.path.join(path, plik)):
            l.append(plik)
    return l

def get_x(r): return f"{path}/{r['ID']}.png"
def get_y(r): return float(r[PREDICTED_VALUE])

path = f'{PATH}/all_qm9'
get_image_files = get_list(path)

f = os.listdir(f"{PATH}/all_qm9")
f = [x[:-4] for x in f]

filtered = ds[ds.ID.isin(f)]
print(f'{len(filtered)} out of {len(ds)} samples were selected')

dblock = DataBlock(blocks=(ImageBlock, RegressionBlock),
                   get_x=get_x, get_y=get_y,
                   splitter=RandomSplitter(valid_pct=0.1, seed=42),
                   item_tfms = Resize(RESIZE)).dataloaders(filtered, bs=BATCH_SIZE)

learn = vision_learner(dblock, resnet18, metrics=mae, lr=LEARNING_RATE)
learn.model.load_state_dict(torch.load(model_path))


#proper files
f = ds.ID.to_list()
f = [f"{x}.png" for x in f]

#validation
test_files = os.listdir(f"{PATH}/all_qm9")
test_files = [x for x in test_files if x in f]
print(len(test_files))
test_data = [PILImage.create(f'{PATH}/all_qm9/{file}') for file in test_files]
test_dl = learn.dls.test_dl(test_data)
preds, _ = learn.get_preds(dl=test_dl)

err = []
for idx in range(1,len(test_files),12):
    print(f'\r{idx}/{len(test_files)}',end='')
    actual = ds[PREDICTED_VALUE].loc[ds.ID == test_files[idx][:-4]].values[0]
    err.append(np.abs(preds[idx].item() - float(actual)))

print(f"\n Average prediction error on test set: {sum(err)/len(err)*27211:.2f} meV")
