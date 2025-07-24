import pandas as pd
import os, sys, random
from scipy.special import factorial
from scripts.params import *
from scripts.utils import get_list_of_files
from ase.data import atomic_numbers

import logging
from scripts.logging import setup_logging

setup_logging()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
    
    #ensuring that atoms are writen in .xyz in correct order
    order_of_atoms = atomic_numbers
    indices = sorted(range(len(atom_type)), key=lambda i: order_of_atoms[atom_type[i]])
    atom_type= [atom_type[i] for i in indices]
    cords = [cords[i] for i in indices]
    mulliken = [mulliken[i] for i in indices]

    def shuffle_atoms(atom_type, cords, mulliken, indices):
        combined = list(zip(atom_type, cords, mulliken))
        random.shuffle(indices)
        shuffled = [combined[i] for i in indices]
        return zip(*shuffled)
    
    if shuffle == 'full':
        indices = list(range(len(atom_type)))  # full randomness
        atom_type, cords, mulliken = shuffle_atoms(atom_type, cords, mulliken, indices)
    
    elif shuffle == 'partial':
        '''Random selection beetwen two groups: CONF and H'''
        heavy_atoms = []
        hydrogen_atoms = []

        for i, atom in enumerate(atom_type):
            if atom == 'H':
                hydrogen_atoms.append((atom_type[i], cords[i], mulliken[i]))
            else:
                heavy_atoms.append((atom_type[i], cords[i], mulliken[i]))

        random.shuffle(heavy_atoms)
        random.shuffle(hydrogen_atoms)

        atom_type = []
        cords = []
        mulliken = []

        for at, crd, mul in heavy_atoms:
            atom_type.append(at)
            cords.append(crd)
            mulliken.append(mul)

        for at, crd, mul in hydrogen_atoms:
            atom_type.append(at)
            cords.append(crd)
            mulliken.append(mul)

    elif shuffle == 'groups':
        '''Shuffling every type of atom separately'''
        atom_groups = {'C': [], 'O': [], 'N': [], 'F': [], 'H': [], 'P':[], 'S':[], 'Cl':[], 'Ru':[], 'Pd':[], 'Pt':[], 'Ir':[], 'B':[]}
        
        for i, atom in enumerate(atom_type):
            atom_groups[atom].append((atom_type[i], cords[i], mulliken[i]))
        
        for atom in atom_groups:
            random.shuffle(atom_groups[atom])
        
        atom_type = []
        cords = []
        mulliken = []
        
        for atom in ['C', 'O', 'N', 'F', 'H']:
            group = atom_groups[atom]
            for at, crd, mul in group:
                atom_type.append(at)
                cords.append(crd)
                mulliken.append(mul)

    '''Putting all together'''
    df_record['atom_type'] = atom_type
    df_record['cords'] = cords
    df_record['mulliken'] = mulliken
    df_record['smiles'] = lines[-2]
    return df_record


def making_df(l:int=0, cycle:int=CYCLE) -> None:
    ''' Function using the extraction funcionality to create an entire database from a list of .xyz file names'''
    files = get_list_of_files()
    
    files = [item for item in files if '.xyz' in item]
        
    print(f"Program found {len(files)} files in data directory")
    if l == 0: l = len(files)
    os.chdir(f'{PATH}/data')
    
    print(f'Creating a database of length {l * CYCLE}')
    chunks = []
    header_bool = True
    for idx, file in enumerate(files):
        print(f'\rProgress: {(idx) / l * 100:.2f}/100', end='')
        df_chunk = pd.DataFrame([extracting(file) for _ in range(cycle)])
        if idx == 1:
            pass
            #columns_ = df_chunk.columns
        chunks.append(df_chunk)
        
        #dividing into chunk procedure
        if len(chunks) > 364000-1:
            print(' Chunk created')
            os.chdir('..')
            pd.concat(chunks).to_csv(f"{PATH}/{DB}.csv", mode='a', header=header_bool, index=False)
            if header_bool == True:
                header_bool = False
            df = pd.read_csv(f"{PATH}/{DB}.csv")
            print(f"Current len of db: {len(df)}")
            chunks = []
            os.chdir(f'{PATH}/data')

    if chunks:
        pd.concat(chunks).to_csv(f"{PATH}/{DB}.csv", mode='a', header=header_bool, index=False)
        
    
    print('\nReading df')
    df = pd.read_csv(f'{PATH}/{DB}.csv')
    print(f'Len of df:{len(df)}')
    #df.columns = columns_

    #counting the atoms 
    el = [x for x in atomic_numbers][:84]

    def count_(l, element):
        return l.count(element)

    for atom in el:
        df[f'Number_of_{atom}'] = df.atom_type.apply(lambda x: count_(x, atom))
        
    df['Sum_of_heavy_atoms'] = df['n_atoms'] - df['Number_of_H']


    #possible comb column creation
    if SHUFFLE == 'none':
        df['possible_comb'] = 1
    elif SHUFFLE == 'partial':
        df['possible_comb'] = factorial(df['Sum_of_heavy_atoms']) * factorial(df['Number_of_H'])
    elif SHUFFLE == 'groups':
        df['possible_comb'] = factorial(df['Number_of_C'])*factorial(df['Number_of_F'])*factorial(df['Number_of_N'])*factorial(df['Number_of_O'])*factorial(df['Number_of_H']) 

    #modify the ID column to avoid overwriting rows in case on CYCLE > 1
    def transform_id(row):
        id_parts = row['ID'].split(" ")
        id_num = id_parts[1]
        index = row.name
        return f"{id_parts[0]}_{id_num}_{index}"

    df['ID']= df.apply(transform_id, axis=1)

    #bandgap rename and transform date type
    df.rename(columns={'HOMO-LUMO Gap': 'bandgap'}, inplace=True)

    df['bandgap'] = df['bandgap'].astype('float32')
    df['bandgap_correct'] = df['bandgap'] - df['bandgap'].mean()
    df['Zero point vibrational energy'] *= 27211

    #optional filtering
    if DB == 'qm7_demo': df = df[df['Sum_of_heavy_atoms']<8]
    if DB == 'qm8_demo': df = df[df['Sum_of_heavy_atoms']<9] 

    #final saving
    df.to_csv(os.path.join(PATH, f"{DB}.csv"))    
    print(f'\nDatabase of lenght {len(df)} was successfully created based on {len(files)} files (Shuffle: {SHUFFLE}, number of data point per molecule: {CYCLE}). Name of db: {PATH}/{DB}.csv')

if __name__ == '__main__':
    making_df()


