import os, random
import re, shutil
import datetime
import importlib
import scripts.params as params_module 
from scripts.params import PATH
from scripts.utils import modify_params


#delete existing .csv file with corresponding names
def remove_csv(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'{file_path} has been removed.')
    else:
        print(f'{file_path} does not exist.')

#remove train, test and data files
def remove_dir(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
        print(f"Directory {directory} have been removed.")

#running run_fastai
def run_script():
    os.system(f'nohup python run_fastai.py > {datetime.datetime.now().strftime("%H_%M_%d_%m_%Y")}.log')

def get_random_params():
    DB = random.choices(['qm9','qm9_2','qm9_3','qm9_4','qm9_6'], weights = [0.1, 0.1, 0.05, 0.12, 0.11])[0]
    BATCH_SIZE = random.choice([x for x in range(16,128,4)])
    TYPE_OF_IMAGE = random.choices(['A','B'], weights = [1,1])[0]
    PATIENCE = random.choice([x for x in range(16,48,4)])

    MODEL_LIST = ['resnet18','vgg19_bn','vgg16_bn','squeezenet1_0', 'squeezenet1_1','S1CNN()','S2CNN()', 'resnet34','resnet50','densenet121']
    
    if DB == 'qm9_6':
        CYCLE = random.choice([8,16,24])
    elif DB == 'qm9_4':
        CYCLE = random.choice([8,16])
    elif DB == 'qm9_3':
        CYCLE = random.choice([8,12])
    elif DB == 'qm9_2':
        CYCLE = random.choice([8])
    elif DB == 'qm9':
        CYCLE = random.choice([4])

    SHUFFLE = 'groups'
    MOMENTUM = round(random.uniform(0.75,0.85),3)
    LEARNING_RATE = round(random.uniform(0.0003,0.003),5)
    MARGIN = random.choice(['avg','black'])


    MODEL = 'vgg19_bn' if TYPE_OF_IMAGE == 'B' else 'S2CNN()'

    RESIZE = random.choice([32,36,40,44,48])
    MATRIX_SIZE = 0
    RANDOM_OR = False

    return {'DB':DB,'BATCH_SIZE':BATCH_SIZE,'CYCLE':CYCLE,'TYPE_OF_IMAGE':TYPE_OF_IMAGE,'SHUFFLE':SHUFFLE,'MOMENTUM':MOMENTUM,
           'LEARNING_RATE':LEARNING_RATE,'MODEL':MODEL,'RESIZE':RESIZE,'MATRIX_SIZE':MATRIX_SIZE,'RANDOM_OR':RANDOM_OR,'MARGIN':MARGIN, 'PATIENCE':PATIENCE }

if __name__ == "__main__":
 
   for _ in range(40):
        os.chdir(PATH)
        modify_params(get_random_params())
        importlib.reload(params_module)

        csv_file = f'{params_module.PATH}/{params_module.DB}.csv'
        remove_csv(csv_file)
        remove_dir('data')
        remove_dir('test')
        remove_dir('train')
        run_script()




    
