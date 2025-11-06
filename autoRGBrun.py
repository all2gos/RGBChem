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

#running run_learning
def run_script():
    print('Preparing for running training procedure')
    os.system(f'nohup python run_learning.py > {datetime.datetime.now().strftime("%H_%M_%d_%m_%Y")}.log')

def get_random_params():
    BATCH_SIZE = random.choice(list(range(24,196,4)))    
    TYPE_OF_IMAGE = random.choices(['A','B', 'G2'], weights = [1,1,3])[0]
    PATIENCE, SHUFFLE = 48, 'groups' #random.choice(list(range(16,48,4)))

    
    if TYPE_OF_IMAGE == 'B':
        MODEL= 'vgg19_bn'
    elif TYPE_OF_IMAGE == 'A':
        MODEL= 'S2CNN()'
    else:
        MODEL = random.choices(['S2CNN()','S1CNN()', 'vgg19_bn', 'resnet18'], weights = [1,1,1,1])[0]
    
    #MODEL = 'vgg19_bn' if TYPE_OF_IMAGE == 'B' else 'S2CNN()'

    #MODEL_LIST = ['resnet18','vgg19_bn','vgg16_bn','squeezenet1_0', 'squeezenet1_1','S1CNN()','S2CNN()', 'resnet34','resnet50','densenet121']
    '''
    if MODEL == 'vgg19_bn':
        DB = random.choices(['qm9_2','qm9','qm9_6'])[0]

        if DB == 'qm9_6':
            CYCLE = 8
        elif DB == 'qm9':
            CYCLE = random.choice([2,4])
        elif DB == 'qm9_2':
            CYCLE = 4

    elif MODEL == 'S2CNN()':
        DB = random.choices(['qm9','qm9_6','qm9_2'])[0]

        if DB == 'qm9_6':
            CYCLE = random.choice([1,2])
        elif DB == 'qm9':
            CYCLE = 2
        elif DB=='qm_2':
            CYCLE = 4
    '''
    DB = random.choices(['qm_4','qm9_6','qm9_2', 'qm9_3'], weights=[1,1,1,1])[0]


    if DB == 'qm9_6':
        CYCLE = random.choice([1,2, 4, 8, 16, 32])

    elif DB=='qm_2':
        CYCLE = random.choice([1,2, 4, 8])
    elif DB == 'qm9_3':
        CYCLE = random.choice([1,2, 4, 8, 16])
    elif DB == 'qm_4':
        CYCLE = random.choice([1,2, 4, 8, 16, 32])
    
    MOMENTUM = round(random.uniform(0.75,0.85),3)
    LEARNING_RATE = round(random.uniform(0.0003,0.003),5)
    MARGIN = random.choice(['avg','black'])

    RESIZE = random.choice([32,36,40,44,48])
    MATRIX_SIZE = 0
    RANDOM_OR = False
    

    params_set = {'DB':DB,'BATCH_SIZE':BATCH_SIZE,'CYCLE':CYCLE,'TYPE_OF_IMAGE':TYPE_OF_IMAGE,'SHUFFLE':SHUFFLE,'MOMENTUM':MOMENTUM,
           'LEARNING_RATE':LEARNING_RATE,'MODEL':MODEL,'RESIZE':RESIZE,'MATRIX_SIZE':MATRIX_SIZE,'RANDOM_OR':RANDOM_OR,'MARGIN':MARGIN, 'PATIENCE':PATIENCE }

    print(params_set)
    return params_set

if __name__ == "__main__":
   
   #for i in ['Dipole moment','bandgap','Zero point vibrational energy','Heat capacity at 298K','Internal energy at 0K']:
    for i in range(20):
        os.chdir(PATH)
        print(f"Running iteration {i+1} of 20")
        modify_params(get_random_params())
        importlib.reload(params_module)

        csv_file = f'{params_module.PATH}/{params_module.DB}.csv'
        remove_csv(csv_file)
        remove_dir('data')
        remove_dir('test')
        remove_dir('train')
        run_script()


        




    
