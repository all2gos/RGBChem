import torch
from torchvision import models 

#--- OVERALL PARAMETERS---#
PATH = '.' #your path to the working directory
PREDICTED_VALUE = 'bandgap_correct' #or 'Energy of HOMO', 'Energy of LUMO' or any other available properties

#---DATABASE PARAMETERS---#
DB = 'qm9' #name of your .csv database
SAMPLE = 1 #the fraction of data from the dataset to be used for training
SHUFFLE = False #enable data augmentation option by randomly selecting the order of coordinate atoms in a molecule, used only when creating a database and not images
CYCLE = 1 #number of images generated to one particle (should be greater than 1 only when coordinate shuffling is enabled)

#---IMAGE PARAMETERS---#

MATRIX_SIZE = 32
TRAIN_DIR_NAME = 'train'
TEST_DIR_NAME = 'test'
COULOMB_DIAGONAL = False # parameter specifying whether values on the diagonal for a coulomb matrix should be displayed

#---TRAINING PROCESS PARAMETERS---#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0003
MOMENTUM = 0.8
EPOCHS = 64
TRAIN_TEST_SPLIT = 0.8 
BATCH_SIZE = 32

#---LOGGING---#
#name of the log file
if SHUFFLE == False:
    LOG_FILE = f'{DB}_e{EPOCHS}_{PREDICTED_VALUE}_bs{BATCH_SIZE}.log' 
else: 
    LOG_FILE = f'{DB}_e{EPOCHS}_{PREDICTED_VALUE}_bs{BATCH_SIZE}_shuffle_{CYCLE}.log' 