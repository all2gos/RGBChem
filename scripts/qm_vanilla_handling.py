import os

from scripts.making_df import *
from scripts.params import *
from scripts.utils import modify_params 

def create_qm_vanilla_file():
    if f"{DB.split('_')[0]}_vanilla.csv" not in os.listdir({PATH}):
        modify_param({'CYCLE':1})
        making_df()
        os.system(f"mv {PATH}/{DB} {PATH}/{DB.split('_')[0]}_vanilla.csv")





