import torch
from torch_geometric.datasets import QM9
from scripts.params import *
import pandas as pd
from ase.data import chemical_symbols
import random

def making_df_tg(CYCLE:int, shuffle=SHUFFLE):
    '''Makes dataframe based on QM9 database from torch_geometric library'''

    path = './data/QM9'
    dataset = QM9(path)

    y_dict = {'bandgap':4}
    #atomic_number_reverse = {v: k for k, v in atomic_numbers.items()}

    dataset_cords = [data.pos for data in dataset]
    dataset_an = [data.z for data in dataset]
    dataset_y = dataset.y[:, y_dict[PREDICTED_VALUE]]

    rows = []
    for molecule in range(len(dataset_cords[:200])):
        for iteration in range(CYCLE):
            row = {'cords': [[f'{coord.item(): .10f}' for coord in atom] for atom in dataset_cords[molecule]],
                'atom_type': [chemical_symbols[int(x)] for x in dataset_an[molecule]],
                f"{PREDICTED_VALUE}":float(dataset_y[molecule])}

            if shuffle != 'groups':
                print('For now only groups shuffling is implemented in torch_geometric dataset')

            '''Shuffling every type of atom separately'''
            atom_groups = {atom: [] for atom in chemical_symbols[:84][::-1]}

            for i, atom in enumerate(row['atom_type']):

                atom_groups[atom].append((row['atom_type'][i], row['cords'][i]))
            
            for atom in atom_groups:
                random.shuffle(atom_groups[atom])

            shuffled_molecule_cords = []
            shuffled_molecule_atom_type = []

            for type_of_atom in atom_groups:
                for atom in atom_groups[type_of_atom]:
                    shuffled_molecule_cords.append(atom[1])
                    shuffled_molecule_atom_type.append(atom[0])

            
            row['atom_type'] = shuffled_molecule_atom_type
            row['cords'] = shuffled_molecule_cords
            row['n_atoms'] = len(row['atom_type'])
            row['ID'] = f'tg_{molecule}_{iteration}'
            rows.append(row)




    df = pd.DataFrame(rows)
    df.to_csv('tg.csv')
    print(f"Dataframe with {len(df)} rows created and saved as 'tg.csv'")


if __name__ == "__main__":
    making_df_tg(CYCLE=CYCLE, shuffle=SHUFFLE)
    print("Dataframe created successfully.")


