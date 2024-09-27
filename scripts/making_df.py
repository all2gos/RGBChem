import random
import pandas as pd
import os, sys
from scipy.special import factorial


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.params import *
from scripts.utils import get_list_of_files

def extracting(f, shuffle = SHUFFLE):
    ''' Extracts information from .xyz file into single dataframe row'''
    with open(f, 'r') as file:
        lines = file.readlines()

    df_record = {'n_atoms': int(lines[0].strip()), 'atom_type': [], 'cords': [], 'mulliken': []}
    properties = lines[1].split("\t")
    

    if COMPRESSION == False:
        labels = ['ID','A','B','C','Dipole moment','Isotropic Polarizability', 'Energy of HOMO',
                'Energy of LUMO','HOMO-LUMO Gap','Electronic spatial extent','Zero point vibrational energy',
                'Internal energy at 0K','Internal energy at 298K','Enthalphy at 298K',
                'Free energy at 298K','Heat capacity at 298K']
    else:
        labels = ['ID',PREDICTED_VALUE]
    
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


def making_df(l:int=0, cycle:int=CYCLE) -> None:
    ''' Function using the extraction funcionality to create an entire database from a list of .xyz file names'''
    files = get_list_of_files()
    if l == 0: l = len(files)
    os.chdir(f'{PATH}/data')
    
    print(f'Creating a database of length {l * CYCLE}')
    chunks = []
    
    for idx, file in enumerate(files):
        if idx % 1000 == 0:
            print(f'\rProgress: {(idx) / l * 100:.2f}/100', end='')
        df_chunk = pd.DataFrame([extracting(file) for _ in range(cycle)])
        chunks.append(df_chunk)
        
        #dividing into chunk procedure
        if len(chunks) > 16000:
            print(' Chunk created')
            os.chdir('..')
            pd.concat(chunks).to_csv(f"{PATH}/{DB}.csv", mode='a', header=not os.path.exists(f"{PATH}/{DB}.csv"))
            chunks = []
            os.chdir(f'{PATH}/data')


    if chunks:
        pd.concat(chunks).to_csv(f"{PATH}/{DB}.csv", mode='a', header=not os.path.exists(f"{PATH}/{DB}.csv"))



    os.chdir('..')

    print('Reading df')
    df = pd.read_csv(f'{PATH}/{DB}.csv')
    print(f'Len of df:{len(df)}')

    el = ['C','H','O','F','N']

    def count_(l, element):
        return l.count(element)

    for atom in el:
        df[f'Number_of_{atom}'] = df.atom_type.apply(lambda x: count_(x, atom))
        
    df['Sum_of_heavy_atoms'] = df['Number_of_C'] + df['Number_of_F'] + df['Number_of_N'] + df['Number_of_O']
    df['possible_comb'] = factorial(df['Sum_of_heavy_atoms']) * factorial(df['Number_of_H'])

    def transform_id(row):
        id_parts = row['ID'].split(" ")
        id_num = id_parts[1]
        index = row.name
        return f"{id_parts[0]}_{id_num}_{index}"

    df['ID'] = df.apply(transform_id, axis=1)

    df.rename(columns={'HOMO-LUMO Gap': 'bandgap'}, inplace=True)

    df['bandgap'] = df['bandgap'].astype('float32')

    df['bandgap_correct'] = df['bandgap'] - df['bandgap'].mean()
    df = df[df['possible_comb'] > 2*CYCLE]

    if DB == 'qm7_demo' :df = df[df['Sum_of_heavy_atoms']<8]
    if DB == 'qm8_demo' :df = df[df['Sum_of_heavy_atoms']<9] 

    df.to_csv(os.path.join(PATH, f"{DB}.csv"))    
    print(f'\nDatabase of lenght {len(df)} was successfully created based on {len(files)} files (Shuffle: {SHUFFLE}, number of data point per molecule: {CYCLE}). Name of db: {PATH}/{DB}.csv')

if __name__ == '__main__':
    making_df()


