import os
import pandas as pd
import random
from scripts.params import CYCLE, SHUFFLE, DB, PATH
from ase.data import atomic_numbers
#from scripts.params import SHUFFLE, CYCLE

def cycle_adjusting(filename, cycle=CYCLE, shuffle=SHUFFLE):
    '''Function which takes .csv database file as input (with at least ID, atom_type and cords columns) 
    and generates new .csv file based on CYCLE and SHUFFLE_TYPE parameter'''


    df = pd.read_csv(filename)

    new_list = []
    for compound in range(len(df[:])):
        for _ in range(CYCLE):
            row = df.iloc[compound]
            l = shuffle_atoms(row.ID, eval(row.atom_type), eval(row.cords), eval(row.mulliken))
            new_list.append(l)
    return pd.DataFrame(new_list, columns=['ID', 'atom_type','cords','mulliken'])
    

def full_shuffle_atoms(atom_type, cords, mulliken, indices):
    combined = list(zip(atom_type, cords, mulliken))
    random.shuffle(indices)
    shuffled = [combined[i] for i in indices]
    return zip(*shuffled)


def shuffle_atoms(ID, atom_type, cords, mulliken, shuffle=SHUFFLE):
    
    if shuffle == 'full':
        indices = list(range(len(atom_type)))  # full randomness
        atom_type, cords, mulliken = full_shuffle_atoms(atom_type, cords, mulliken, indices)
    
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
        atom_groups = {}
        
        for i, atom in enumerate(atom_type):
            if atom not in atom_groups:
                atom_groups[atom] = []

            atom_groups[atom].append((atom_type[i], cords[i], mulliken[i]))
        
        for atom in atom_groups.keys():
            random.shuffle(atom_groups[atom])
        
        atom_type, cords, mulliken = [], [], []

        
        at_num_order = sorted(atom_groups.keys(), key=lambda x: atomic_numbers[x], reverse=True)
        for atom in at_num_order:
            group = atom_groups[atom]
            for at, crd, mul in group:
                atom_type.append(at)
                cords.append(crd)
                mulliken.append(mul)

    return ID, atom_type, cords, mulliken
    


def cycle_adjusting_main(DB=DB):
    '''Main function to run cycle adjusting'''
    df = pd.read_csv(f'{PATH}/{DB}.csv')
    r = cycle_adjusting(f'{PATH}/{DB}.csv')
    r = r.merge(df[['bandgap','ID','n_atoms']], on = 'ID')
    r['ID'] = r['ID'] + '_' + r.index.astype(str)
    r.to_csv(f'{DB}.csv', index=False)
    print(f"Created {DB}.csv with {len(r)} rows")
    return r


