import random
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.params import *
from scripts.utils import get_list_of_files

def extracting(f, shuffle = SHUFFLE):
    ''' Extracts information from .xyz file into single dataframe row'''
    with open(f, 'r') as file:
        lines = file.readlines()

    df_record = {'n_atoms': int(lines[0].strip()), 'atom_type': [], 'cords': [], 'mulliken': []}
    properties = lines[1].split("\t")
    
    labels = ['ID','A','B','C','Dipole moment','Isotropic Polarizability', 'Energy of HOMO',
              'Energy of LUMO','HOMO-LUMO Gap','Electronic spatial extent','Zero point vibrational energy',
              'Internal energy at 0K','Internal energy at 298K','Enthalphy at 298K',
              'Free energy at 298K','Heat capacity at 298K']
    
    df_record.update(zip(labels, properties))
    
    #coordinates 
    atom_type, cords, mulliken = [],[],[]
    for atom in range(df_record['n_atoms']):
        atom_record = lines[2+atom].replace('\n','').split('\t')
        atom_type.append(atom_record[0])
        cords.append(atom_record[1:-1])
        mulliken.append(atom_record[-1])

    p = df_record['n_atoms'] - atom_type.count('H')
    '''If shuffle = full or partial then the order of atoms in the molecule is randomized'''
    if shuffle == 'full':
        combined = list(zip(atom_type, cords, mulliken))
        random.shuffle(combined)
        atom_type, cords, mulliken = zip(*combined)
    elif shuffle == 'partial':
        heavy_atoms = list(zip(atom_type[:p], cords[:p],mulliken[:p]))
        hydrogen = list(zip(atom_type[p:],cords[p:],mulliken[p:]))
        random.shuffle(heavy_atoms)
        random.shuffle(hydrogen)

        atom_type_heavy, cords_heavy, mulliken_heavy = zip(*heavy_atoms) 

        if len(hydrogen) != 0:
            atom_type_h, cords_h, mulliken_h = zip(*hydrogen)

            atom_type, cords, mulliken = atom_type_heavy+atom_type_h, cords_heavy+cords_h, mulliken_heavy+mulliken_h
        else:
            atom_type, cords, mulliken = atom_type_heavy, cords_heavy, mulliken_heavy
    '''Putting all together'''
    df_record['atom_type'] = atom_type
    df_record['cords'] = cords
    df_record['mulliken'] = mulliken
    return df_record

def making_df(l:int=0, cycle:int=CYCLE) -> pd.DataFrame:
    ''' Function using the extraction funcionality to create an entire database from a list of .xyz file names'''
    df = []
    files = get_list_of_files()
    if l==0: l=len(files) 
    os.chdir(f'{PATH}/data')
    
    #random_files = random.sample(files, l)
    print(f'Creating a database of length {l*CYCLE}')
    for idx, file in enumerate(files):
        if idx % 1000 == 0:
            print(f'\rProgress: {(idx+883) / l*100:.2f}/100',end='')
        df.extend(extracting(file) for _ in range(cycle))


    df = pd.DataFrame(data=df)

    el = ['C','H','O','F','N']

    def count_(l, element):
        return l.count(element)

    for atom in el:
        df[f'Number_of_{atom}'] = df.atom_type.apply(lambda x: count_(x, atom))
        
    df['Sum_of_heavy_atoms'] = df['Number_of_C'] + df['Number_of_F'] + df['Number_of_N'] + df['Number_of_O']

    def transform_id(row):
        id_parts = row['ID'].split(" ")
        id_num = id_parts[1]
        index = row.name
        return f"{id_parts[0]}_{id_num}_{index}"

    df['ID'] = df.apply(transform_id, axis=1)

    df.rename(columns={'HOMO-LUMO Gap': 'bandgap'}, inplace=True)

    df['bandgap'] = df['bandgap'].astype('float32')

    df['bandgap_correct'] = df['bandgap'] - df['bandgap'].mean()
    os.chdir('..')

    correct_order = ['ID','A','B','C','Dipole moment','Isotropic Polarizability', 'Energy of HOMO',
              'Energy of LUMO','bandgap','bandgap_correct','Electronic spatial extent','Zero point vibrational energy',
              'Internal energy at 0K','Internal energy at 298K','Enthalphy at 298K',
              'Free energy at 298K','Heat capacity at 298K', 'n_atoms','atom_type','cords','mulliken', 'Number_of_C','Number_of_F','Number_of_N','Number_of_O','Number_of_H','Sum_of_heavy_atoms']
    
    df = df[correct_order]

    if DB == 'qm7_demo' :df = df[df['Sum_of_heavy_atoms']<8] 
    df.to_csv(os.path.join(PATH, f"{DB}.csv"))    
    print(f'\nDatabase of lenght {len(df)} was successfully created based on {len(files)} files (Shuffle: {SHUFFLE}, number of data point per molecule: {CYCLE})')
    print(f'Database was saved as {PATH}/{DB}.csv"')
    return df

if __name__ == '__main__':
    making_df()
