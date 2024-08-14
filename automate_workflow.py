import os
import re
import time
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
    os.system(f'nohup python run_fastai.py > {int(time.time())}.log')


# Przykład użycia:
if __name__ == "__main__":

    changes = {
        'EPOCHS': 8,
    }
    modify_params(changes)
    importlib.reload(params_module)

    csv_file = f'{params_module.DB}.csv'
    #remove_csv(csv_file)
    run_script()


    changes = {
        'EPOCHS': 16,
    }
    modify_params(changes)
    importlib.reload(params_module)

    csv_file = f'{params_module.DB}.csv'
    #remove_csv(csv_file)
    run_script()
