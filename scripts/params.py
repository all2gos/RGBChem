import torch
from torchvision import models 

__all__ = ['BATCH_SIZE', 'CYCLE', 'DB', 'DEVICE', 'EPOCHS', 'LEARNING_RATE', 'LOG_FILE', 'MATRIX_SIZE', 'MOMENTUM', 'PATH', 'PREDICTED_VALUE', 'SAMPLE', 'SHUFFLE', 'TEST_DIR_NAME',
 'TRAIN_DIR_NAME', 'TRAIN_TEST_SPLIT', 'SCALING', 'DELETE','TYPE_OF_IMAGE', 'PATIENCE', 'DELTA','MODEL','RANDOM_OR','MARGIN','RESIZE', 'COMPRESSION','STEP']

#--- OVERALL PARAMETERS---#
PATH = '.' #your path to the working directory
PREDICTED_VALUE = 'bandgap' #or 'Energy of HOMO', 'Energy of LUMO' or any other available properties

#---DATABASE PARAMETERS---#
DB = 'qm9'
SAMPLE = 1 #the fraction of data from the dataset to be used for training
CYCLE = 1
SHUFFLE = 'none' #none, full, partial, groups
#none = defined order of atoms
#full = completely random order
#partial = divide atoms into two groups (H and not H) and randomly selection in these groups
#groups = similar to partial but every type of atom is a separate group
COMPRESSION = False #if True then all dataset is smaller

#---IMAGE PARAMETERS---#
MATRIX_SIZE = 0 #if zero then images doee not have margins
TRAIN_DIR_NAME = 'train'
TEST_DIR_NAME = 'test'
TYPE_OF_IMAGE = 'H'
RANDOM_OR = False #if True then images will be placed randomly
SCALING = True
DELETE = True  #if True then the script will delete all so far generated files and created new one from scratch
MARGIN = 'avg' #the way the margins are filled
RESIZE = 32
STEP = 60 #calibration parameter
#---TRAINING PROCESS PARAMETERS---#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = 'resnet18'
LEARNING_RATE = '0.0003'
MOMENTUM = 0.9
EPOCHS = 512
TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 256

#---EARLY STOPPING---#
PATIENCE = 20
DELTA = 0

#---LOGGING---#
#name of the log file
if SHUFFLE not in ['partial','full']:
    LOG_FILE = f'{DB}{TYPE_OF_IMAGE}{MODEL}_{MARGIN}_e{EPOCHS}_{PREDICTED_VALUE}_bs{BATCH_SIZE}_size{RESIZE}.log' 
else: 
    LOG_FILE = f'{DB}{TYPE_OF_IMAGE}{MODEL}_{MARGIN}_e{EPOCHS}_{PREDICTED_VALUE}_bs{BATCH_SIZE}_size{RESIZE}_shuffle_{SHUFFLE}_{CYCLE}.log' 
