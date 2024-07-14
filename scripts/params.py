import torch
from torchvision import models 

__all__ = ['BATCH_SIZE', 'CYCLE', 'DB', 'DEVICE', 'EPOCHS', 'LEARNING_RATE', 'LOG_FILE', 'MATRIX_SIZE', 'MOMENTUM', 'PATH', 'PREDICTED_VALUE', 'SAMPLE', 'SHUFFLE', 'TEST_DIR_NAME',
 'TRAIN_DIR_NAME', 'TRAIN_TEST_SPLIT', 'SCALING', 'DELETE','TYPE_OF_IMAGE', 'PATIENCE', 'DELTA','MODEL','RANDOM_OR','MARGIN','RESIZE']

#--- OVERALL PARAMETERS---#
PATH = '.' #your path to the working directory
PREDICTED_VALUE = 'bandgap' #or 'Energy of HOMO', 'Energy of LUMO' or any other available properties

#---DATABASE PARAMETERS---#
DB = 'qm7_demo' #name of your .csv database
SAMPLE = 1 #the fraction of data from the dataset to be used for training
CYCLE = 1 #number of images generated to one particle (should be greater than 1 only when coordinate shuffling is enabled)
SHUFFLE = 'partial' #shuffle is a complex parameter: when it is set to full then each time order of the atoms in particle is randomly selected,
#when it is set to partial then the algorithm split atoms in particle into two groups: heavy atoms and hydrogens, and randomly selected order 
#inside these particular group
#when it is set to anything else then shuffle will not be perform


#---IMAGE PARAMETERS---#
MATRIX_SIZE = 28 #if zero then images doee not have margins
TRAIN_DIR_NAME = 'train'
TEST_DIR_NAME = 'test'
TYPE_OF_IMAGE = 'A' #or any other letter available 
RANDOM_OR = False #if True then images will be placed randomly
SCALING = True
DELETE = True #if True then the script will delete all so far generated files and created new one from scratch
MARGIN = 'avg' #the way the margins are filled
RESIZE = 23 
#---TRAINING PROCESS PARAMETERS---#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = 'resnet18'
LEARNING_RATE = 0.0003
MOMENTUM = 0.9
EPOCHS = 512
TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 64

#---EARLY STOPPING---#
PATIENCE = 20
DELTA = 0

#---LOGGING---#
#name of the log file
if SHUFFLE == False:
    LOG_FILE = f'{DB}{TYPE_OF_IMAGE}{MODEL}_{MARGIN}_e{EPOCHS}_{PREDICTED_VALUE}_bs{BATCH_SIZE}.log' 
else: 
    LOG_FILE = f'{DB}{TYPE_OF_IMAGE}{MODEL}_{MARGIN}_e{EPOCHS}_{PREDICTED_VALUE}_bs{BATCH_SIZE}_shuffle_{CYCLE}.log' 
