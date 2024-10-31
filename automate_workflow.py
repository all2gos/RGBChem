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
    DB = random.choice(['qm9','qm9_2','qm9_3','qm9_4','qm9_6'])
    BATCH_SIZE = random.choice([x for x in range(16,64,4)])
    TYPE_OF_IMAGE = random.choice(['A','B','C','D','E','F','G','H'])
    PATIENCE = random.choice([x for x in range(20,41,5)])


    if DB == 'qm9_6':
        CYCLE = random.choice([2,4,8,16])
    elif DB == 'qm9_4':
        CYCLE = random.choice([2,4,8])
    elif DB == 'qm9_3':
        CYCLE = random.choice([1,2,4,8])
    elif DB == 'qm9_2':
        CYCLE = random.choice([1,2,4])
    elif DB == 'qm9':
        CYCLE = random.choice([1,2])

    SHUFFLE = random.choices(['full','none','partial','groups'], weights = [3,3,3,3])[0]
    MOMENTUM = round(random.uniform(0.75,0.85),3)
    LEARNING_RATE = round(random.uniform(0.0003,0.003),5)
    MARGIN = random.choice(['avg','black'])
    if DB not in ['qm9','qm9_2']:
        MODEL = random.choice(['resnet18','resnet34','resnet50','vgg16_bn', 'vgg19_bn','densenet121','squeezenet1_0', 'squeezenet1_1'])
    else:
        MODEL = random.choice(['resnet18','resnet34','vgg16_bn', 'vgg19_bn','densenet121','squeezenet1_0', 'squeezenet1_1'])

    if RESIZE_BOOL := random.choices([0, 1], weights=[4, 1])[0]:
        RESIZE = random.choice([32,36,40,44,48])
        MATRIX_SIZE = 0
        RANDOM_OR = False
        
    else:
        MATRIX_SIZE = random.choice([32,36,40,44,48])
        RANDOM_OR = random.choices([False, True], weights = [1,4])[0]
        RESIZE = 0

    return {'DB':DB,'BATCH_SIZE':BATCH_SIZE,'CYCLE':CYCLE,'TYPE_OF_IMAGE':TYPE_OF_IMAGE,'SHUFFLE':SHUFFLE,'MOMENTUM':MOMENTUM,
           'LEARNING_RATE':LEARNING_RATE,'MODEL':MODEL,'RESIZE':RESIZE,'MATRIX_SIZE':MATRIX_SIZE,'RANDOM_OR':RANDOM_OR,'MARGIN':MARGIN}
# Przykład użycia:
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

'''

    {'DB': 'qm9_32', 'BATCH_SIZE': 96, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'D'},
    {'DB': 'qm8_demo', 'BATCH_SIZE': 256, 'CYCLE': 32, 'TYPE_OF_IMAGE': 'E'},
    {'DB': 'qm7_demo', 'BATCH_SIZE': 128, 'CYCLE': 64, 'TYPE_OF_IMAGE': 'G'},
    {'DB': 'qm8_demo', 'BATCH_SIZE': 64, 'CYCLE': 64, 'TYPE_OF_IMAGE': 'H'},
    {'DB': 'qm7_demo', 'BATCH_SIZE': 256, 'CYCLE': 128, 'TYPE_OF_IMAGE': 'H'},
    {'DB': 'qm8_demo', 'BATCH_SIZE': 96, 'CYCLE': 128, 'TYPE_OF_IMAGE': 'G'},
    {'DB': 'qm9_32', 'BATCH_SIZE': 64, 'CYCLE': 32, 'TYPE_OF_IMAGE': 'C'},
    {'DB': 'qm9_32', 'BATCH_SIZE': 256, 'CYCLE': 64, 'TYPE_OF_IMAGE': 'B'},
    {'DB': 'qm9_32', 'BATCH_SIZE': 128, 'CYCLE': 128, 'TYPE_OF_IMAGE': 'A'},
    {'DB': 'qm9_6', 'BATCH_SIZE': 16, 'CYCLE': 8, 'TYPE_OF_IMAGE': 'F'},
    {'DB': 'qm9_6', 'BATCH_SIZE': 256, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'C'},
    {'DB': 'qm9_6', 'BATCH_SIZE': 128, 'CYCLE': 32, 'TYPE_OF_IMAGE': 'D'},
    {'DB': 'qm9_6', 'BATCH_SIZE': 96, 'CYCLE': 64, 'TYPE_OF_IMAGE': 'A'},
    {'DB': 'qm9_6', 'BATCH_SIZE': 64, 'CYCLE': 128, 'TYPE_OF_IMAGE': 'B'},
        

    {'DB': 'qm9_4', 'BATCH_SIZE': 48, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'B'},
    {'DB': 'qm9_4', 'BATCH_SIZE': 32, 'CYCLE': 32, 'TYPE_OF_IMAGE': 'A'},
    {'DB': 'qm9_4', 'BATCH_SIZE': 24, 'CYCLE': 64, 'TYPE_OF_IMAGE': 'D'},
    {'DB': 'qm9_4', 'BATCH_SIZE': 16, 'CYCLE': 128, 'TYPE_OF_IMAGE': 'C'},
    {'DB': 'qm9_3', 'BATCH_SIZE': 24, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'A'},
    {'DB': 'qm9_3', 'BATCH_SIZE': 16, 'CYCLE': 32, 'TYPE_OF_IMAGE': 'B'},
    {'DB': 'qm9_3', 'BATCH_SIZE': 48, 'CYCLE': 64, 'TYPE_OF_IMAGE': 'C'},
    {'DB': 'qm9_3', 'BATCH_SIZE': 32, 'CYCLE': 128, 'TYPE_OF_IMAGE': 'D'},
    {'DB': 'qm9_2', 'BATCH_SIZE': 16, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'H'},
    {'DB': 'qm9_2', 'BATCH_SIZE': 24, 'CYCLE': 32, 'TYPE_OF_IMAGE': 'G'},
    {'DB': 'qm9_2', 'BATCH_SIZE': 32, 'CYCLE': 64, 'TYPE_OF_IMAGE': 'F'},
    {'DB': 'qm9_2', 'BATCH_SIZE': 48, 'CYCLE': 128, 'TYPE_OF_IMAGE': 'E'},



    {'DB': 'qm9', 'BATCH_SIZE': 96, 'CYCLE': 8, 'TYPE_OF_IMAGE': 'B'},
    {'DB': 'qm9', 'BATCH_SIZE': 32, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'G'},
    {'DB': 'qm9', 'BATCH_SIZE': 48, 'CYCLE': 32, 'TYPE_OF_IMAGE': 'H'},
    {'DB': 'qm9', 'BATCH_SIZE': 16, 'CYCLE': 64, 'TYPE_OF_IMAGE': 'E'},
    {'DB': 'qm9', 'BATCH_SIZE': 24, 'CYCLE': 128, 'TYPE_OF_IMAGE': 'F'},
    
    {'DB': 'qm9', 'BATCH_SIZE': 16, 'CYCLE': 8, 'TYPE_OF_IMAGE': 'A'},
    {'DB': 'qm9_6', 'BATCH_SIZE': 256, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'A'},
    {'DB': 'qm9_3', 'BATCH_SIZE': 16, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'A'},
    {'DB': 'qm9', 'BATCH_SIZE': 32, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'A'},

    {'DB': 'qm9_4', 'BATCH_SIZE': 256, 'CYCLE': 1, 'TYPE_OF_IMAGE': 'F'}] #batch_size spory
'''
    


    
