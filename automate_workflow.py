import os
import re
import datetime
import importlib
import scripts.params as params_module 


def modify_params(changes):
    with open('scripts/params.py', 'r') as file:
        content = file.read()

    for param, value in changes.items():
        if isinstance(value, str):
            value = f"'{value}'"
        content = re.sub(f'{param} = .*', f'{param} = {value}', content)


    with open('scripts/params.py', 'w') as file:
        file.write(content)

def remove_csv(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'{file_path} has been removed.')
    else:
        print(f'{file_path} does not exist.')

# 3. Uruchomienie pliku run_fastai.py
def run_script():
    os.system(f'nohup python run_fastai.py > {datetime.datetime.now().strftime("%H_%M_%d_%m_%Y")}.log')


# Przykład użycia:
if __name__ == "__main__":
 
    #cycle shuffle i base

    #{'DB': 'qm7_demo', 'BATCH_SIZE': 16, 'CYCLE': 1, 'TYPE_OF_IMAGE': 'A'},
    #{'DB': 'qm8_demo', 'BATCH_SIZE': 32, 'CYCLE': 1, 'TYPE_OF_IMAGE': 'B'},
    #{'DB': 'qm9_32', 'BATCH_SIZE': 24, 'CYCLE': 1, 'TYPE_OF_IMAGE': 'H'},
    #{'DB': 'qm9_6', 'BATCH_SIZE': 48, 'CYCLE': 1, 'TYPE_OF_IMAGE': 'G'},
        
    trainings = [
    #{'DB': 'qm9_3', 'BATCH_SIZE': 96, 'CYCLE': 1, 'TYPE_OF_IMAGE': 'E'},
    #{'DB': 'qm9_2', 'BATCH_SIZE': 64, 'CYCLE': 1, 'TYPE_OF_IMAGE': 'D'},
    #{'DB': 'qm9', 'BATCH_SIZE': 128, 'CYCLE': 1, 'TYPE_OF_IMAGE': 'C'},
    #{'DB': 'qm7_demo', 'BATCH_SIZE': 24, 'CYCLE': 2, 'TYPE_OF_IMAGE': 'B'},
    #{'DB': 'qm8_demo', 'BATCH_SIZE': 48, 'CYCLE': 2, 'TYPE_OF_IMAGE': 'A'},
    #{'DB': 'qm9_32', 'BATCH_SIZE': 16, 'CYCLE': 2, 'TYPE_OF_IMAGE': 'G'},
    #{'DB': 'qm9_6', 'BATCH_SIZE': 32, 'CYCLE': 2, 'TYPE_OF_IMAGE': 'H'},
    #{'DB': 'qm9_4', 'BATCH_SIZE': 128, 'CYCLE': 2, 'TYPE_OF_IMAGE': 'E'},
    #{'DB': 'qm9_3', 'BATCH_SIZE': 64, 'CYCLE': 2, 'TYPE_OF_IMAGE': 'F'},
    #{'DB': 'qm9_2', 'BATCH_SIZE': 96, 'CYCLE': 2, 'TYPE_OF_IMAGE': 'C'},
    #{'DB': 'qm9', 'BATCH_SIZE': 256, 'CYCLE': 2, 'TYPE_OF_IMAGE': 'D'},
    #{'DB': 'qm7_demo', 'BATCH_SIZE': 32, 'CYCLE': 4, 'TYPE_OF_IMAGE': 'C'},
    #{'DB': 'qm8_demo', 'BATCH_SIZE': 16, 'CYCLE': 4, 'TYPE_OF_IMAGE': 'D'},
    #{'DB': 'qm9_32', 'BATCH_SIZE': 48, 'CYCLE': 4, 'TYPE_OF_IMAGE': 'F'},
    #{'DB': 'qm9_6', 'BATCH_SIZE': 24, 'CYCLE': 4, 'TYPE_OF_IMAGE': 'E'},
    #{'DB': 'qm9_4', 'BATCH_SIZE': 96, 'CYCLE': 4, 'TYPE_OF_IMAGE': 'H'},
    #{'DB': 'qm9_3', 'BATCH_SIZE': 256, 'CYCLE': 4, 'TYPE_OF_IMAGE': 'G'},
    #{'DB': 'qm9_2', 'BATCH_SIZE': 128, 'CYCLE': 4, 'TYPE_OF_IMAGE': 'B'},
    #{'DB': 'qm9', 'BATCH_SIZE': 64, 'CYCLE': 4, 'TYPE_OF_IMAGE': 'A'},
    #{'DB': 'qm7_demo', 'BATCH_SIZE': 48, 'CYCLE': 8, 'TYPE_OF_IMAGE': 'D'},
    #{'DB': 'qm8_demo', 'BATCH_SIZE': 24, 'CYCLE': 8, 'TYPE_OF_IMAGE': 'C'},
    #{'DB': 'qm9_32', 'BATCH_SIZE': 32, 'CYCLE': 8, 'TYPE_OF_IMAGE': 'E'},
    #{'DB': 'qm9_4', 'BATCH_SIZE': 64, 'CYCLE': 8, 'TYPE_OF_IMAGE': 'G'},
    #{'DB': 'qm9_3', 'BATCH_SIZE': 128, 'CYCLE': 8, 'TYPE_OF_IMAGE': 'H'},
    #{'DB': 'qm9_2', 'BATCH_SIZE': 256, 'CYCLE': 8, 'TYPE_OF_IMAGE': 'A'},
    #{'DB': 'qm7_demo', 'BATCH_SIZE': 64, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'E'},
    #{'DB': 'qm8_demo', 'BATCH_SIZE': 128, 'CYCLE': 16, 'TYPE_OF_IMAGE': 'F'},
    #{'DB': 'qm7_demo', 'BATCH_SIZE': 96, 'CYCLE': 32, 'TYPE_OF_IMAGE': 'F'},
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

    

    for training in trainings:
        modify_params(training)
        importlib.reload(params_module)
        csv_file = f'{params_module.DB}.csv'
        remove_csv(csv_file)
        run_script()

    
