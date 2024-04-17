MATRIX_SIZE = 32
PATH = '' #your path to the working directory
DB = 'qm7' #or qm7
PREDICTED_VALUE = 'HOMO-LUMO Gap' #or 'Energy of HOMO', 'Energy of LUMO' or any other available properties

TRAIN_DIR_NAME = 'train'
TEST_DIR_NAME = 'test'
SAMPLE = 1 #the fraction of data from the dataset to be used for training

SHUFFLE = True #enable data augmentation option by randomly selecting the order of coordinate atoms in a molecule
CYCLE = 3 #number of images generated to one particle (should be greater than 1 only when coordinate shuffling is enabled)

COULOMB_DIAGONAL = False # parameter specifying whether values on the diagonal for a coulomb matrix should be displayed
