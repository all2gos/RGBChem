import torch
from torchvision import models 

__all__ = ['BATCH_SIZE', 'CYCLE', 'DB', 'DEVICE', 'EPOCHS', 'LEARNING_RATE', 'LOG_FILE', 'MATRIX_SIZE', 'MOMENTUM', 'PATH', 'PREDICTED_VALUE', 'SHUFFLE', 'TEST_DIR_NAME','TRAIN_DIR_NAME', 'TRAIN_TEST_SPLIT', 'SCALING', 'DELETE','TYPE_OF_IMAGE', 'PATIENCE', 'DELTA','MODEL','RANDOM_OR','MARGIN','RESIZE','STEP','BATTERY_LEVEL_CONTROL','MULTIPROCESS','NUM_PROC','DROPOUT']

#--- OVERALL PARAMETERS---#
PATH = '/home/rstottko/RGBChem' #your path to the working directory
PREDICTED_VALUE = 'bandgap' #or 'Energy of HOMO', 'Energy of LUMO' or any other available properties

#---DATABASE PARAMETERS---#
DB = 'qm9'
CYCLE = 2
SHUFFLE = 'groups'
#none = defined order of atoms
#full = completely random order
#partial = divide atoms into two groups (H and not H) and randomly selection in these groups
#groups = similar to partial but every type of atom is a separate group

#---IMAGE PARAMETERS---#
MATRIX_SIZE = 40
TRAIN_DIR_NAME = 'train'
TEST_DIR_NAME = 'test'
TYPE_OF_IMAGE = 'D'
RANDOM_OR = False
SCALING = True
DELETE = True #if True then the script will delete all so far generated files and created new one from scratch
MARGIN = 'black'
RESIZE = 0
STEP = 10 #calibration parameter
#---TRAINING PROCESS PARAMETERS---#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = 'vgg16_bn'
DROPOUT = 0.2 #only for SnCNN architecture type
LEARNING_RATE = 0.00181
MOMENTUM = 0.81
EPOCHS = 2048
TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 60

#---EARLY STOPPING---#
PATIENCE = 44
DELTA = 0

#---OTHERS---#
BATTERY_LEVEL_CONTROL = True #if True then script checks the battery charge level after each epoch and wait some time  
MULTIPROCESS = True
NUM_PROC=6
#---LOGGING---#
#name of the log file
LOG_FILE = f"{DB}{TYPE_OF_IMAGE}{MODEL}_{MARGIN if MATRIX_SIZE != 0 else 'resize'}_e{EPOCHS}_{PREDICTED_VALUE}_bs{BATCH_SIZE}_size{max(RESIZE,MATRIX_SIZE)}_shuffle_{SHUFFLE}_{CYCLE}.log" 
