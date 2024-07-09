import torch
from torchvision import models 

__all__ = ['BATCH_SIZE', 'CYCLE', 'DB', 'DEVICE', 'EPOCHS', 'LEARNING_RATE', 'LOG_FILE', 'MATRIX_SIZE', 'MOMENTUM', 'PATH', 'PREDICTED_VALUE', 'SAMPLE', 'SHUFFLE', 'TEST_DIR_NAME',
 'TRAIN_DIR_NAME', 'TRAIN_TEST_SPLIT', 'SCALING', 'DELETE','TYPE_OF_IMAGE', 'PATIENCE', 'DELTA','MODEL','RANDOM_OR']

#--- OVERALL PARAMETERS---#
PATH = '.' #your path to the working directory
PREDICTED_VALUE = 'bandgap' #or 'Energy of HOMO', 'Energy of LUMO' or any other available properties

#---DATABASE PARAMETERS---#
DB = 'qm9' #name of your .csv database
SAMPLE = 1 #the fraction of data from the dataset to be used for training
SHUFFLE = False #enable data augmentation option by randomly selecting the order of coordinate atoms in a molecule, used only when creating a database and not images
CYCLE = 1 #number of images generated to one particle (should be greater than 1 only when coordinate shuffling is enabled)

#---IMAGE PARAMETERS---#
MATRIX_SIZE = 32
TRAIN_DIR_NAME = 'train'
TEST_DIR_NAME = 'test'
TYPE_OF_IMAGE = 'A' #or any other letter available 
RANDOM_OR = True #if True then images will be placed randomly
SCALING = True
DELETE = True #if True then the script will delete all so far generated files and created new one from scratch

#---TRAINING PROCESS PARAMETERS---#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = 'resnet18'
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 512
TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 64

#---EARLY STOPPING---#
PATIENCE = 10
DELTA = 0

#---LOGGING---#
#name of the log file
if SHUFFLE == False:
    LOG_FILE = f'{DB}{TYPE_OF_IMAGE}{MODEL}_e{EPOCHS}_{PREDICTED_VALUE}_bs{BATCH_SIZE}.log' 
else: 
    LOG_FILE = f'{DB}{TYPE_OF_IMAGE}{MODEL}_e{EPOCHS}_{PREDICTED_VALUE}_bs{BATCH_SIZE}_shuffle_{CYCLE}.log' 
