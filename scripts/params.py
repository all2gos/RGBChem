import torch

__all__ = ['BATCH_SIZE', 'CYCLE', 'DB', 'DEVICE', 'EPOCHS', 'LEARNING_RATE', 'LOG_FILE', 'MATRIX_SIZE', 'MOMENTUM', 'PATH', 'PREDICTED_VALUE', 'SHUFFLE', 'TEST_DIR_NAME','TRAIN_DIR_NAME', 'TRAIN_VAL_SPLIT', 'SCALING', 'DELETE','TYPE_OF_IMAGE', 'PATIENCE', 'DELTA','MODEL','RANDOM_OR','MARGIN','RESIZE','STEP','BATTERY_LEVEL_CONTROL','MULTIPROCESS','NUM_PROC','DROPOUT', 'WAIT_UNTIL','LOG_LEVEL']

#--- OVERALL PARAMETERS---#
PATH = '/home/rstottko/a_phd_workspace/RGBChem'
PREDICTED_VALUE = 'HL_Gap'

#---DATABASE PARAMETERS---#
DB = 'qm9_6'
CYCLE = 1
SHUFFLE = 'groups'
#none = defined order of atoms
#full = completely random order
#partial = divide atoms into two groups (H and not H) and randomly selection in these groups
#groups = similar to partial but every type of atom is a separate group

#---IMAGE PARAMETERS---#
MATRIX_SIZE = 0
TRAIN_DIR_NAME = 'train'
TEST_DIR_NAME = 'test'
TYPE_OF_IMAGE = 'A'
RANDOM_OR = False
SCALING = True
DELETE = True
MARGIN = 'resize'
RESIZE = 74
STEP = 100 #calibration parameter

#---TRAINING PROCESS PARAMETERS---#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = 'S2CNN()'
DROPOUT = 0.2 #only for SnCNN architecture type
LEARNING_RATE = 0.00245
MOMENTUM = 0.778
EPOCHS = 8
TRAIN_VAL_SPLIT = 0.9
BATCH_SIZE = 100

#---EARLY STOPPING---#
PATIENCE = 2
DELTA = 0
WAIT_UNTIL = 10 #early stopping starts after this many epochs

#---OTHERS---#
BATTERY_LEVEL_CONTROL = False #if True then script checks the battery charge level after each epoch and wait some time  

#---MULTIPROCESSING---#
MULTIPROCESS = True
NUM_PROC=12

#---LOGGING---#
#name of the log file
LOG_LEVEL = 'INFO' #DEBUG, INFO, WARNING, ERROR or CRITICAL
LOG_FILE = f"{DB}{TYPE_OF_IMAGE}{MODEL}_{MARGIN if MATRIX_SIZE != 0 else 'resize'}_e{EPOCHS}_{PREDICTED_VALUE}_bs{BATCH_SIZE}_size{max(RESIZE,MATRIX_SIZE)}_shuffle_{SHUFFLE}_{CYCLE}.log" 
