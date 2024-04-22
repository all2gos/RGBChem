import os
import pandas as pd
import numpy as np
from matrix_function import *
from PIL import Image
import random
from pathlib import Path
from params import *
import random

files = os.listdir(f'{PATH}/data')

#extracts information from .xyz files into df
def extracting(f, shuffle = SHUFFLE):
    with open(f, 'r') as file:
        lines = file.readlines()

    df_record = {'n_atoms':int(lines[0].strip())}

    properties = lines[1].split("\t")
    
    labels = ['ID','A','B','C','Dipole moment','Isotropic Polarizability', 'Energy of HOMO',
              'Energy of LUMO','HOMO-LUMO Gap','Electronic spatial extent','Zero point vibrational energy',
              'Internal energy at 0K','Internal energy at 298K','Enthalphy at 298K',
              'Free energy at 298K','Heat capacity at 298K']
    
    df_record.update(zip(labels, properties))

    #coordinates 
    
    cords = []
    mulliken = []
    atom_type = []

    for atom in range(df_record['n_atoms']):
        temp = lines[2+atom].replace('\n','').split('\t')
        atom_type.append(temp[0])
        cords.append(temp[1:-1])
        mulliken.append(temp[-1])

    if shuffle == True:
        combined = list(zip(atom_type, cords, mulliken))
        random.shuffle(combined)
        atom_type, cords, mulliken = zip(*combined)


    df_record['atom_type'] = atom_type
    df_record['cords'] = cords
    df_record['mulliken'] = mulliken


    return df_record

def making_df(l=len(files), cycle=CYCLE):
    df = []

    os.chdir(f'{PATH}/data')

    random_file = random.sample(files, l)

    for idx, file in enumerate(random_file):
        if idx % 1000 == 0:
            print(round(idx / l * 100, 2))
        for c in range(cycle):
            df.append(extracting(file))

        
    ds = pd.DataFrame(data = df)

    
    print(f'Database of lenght {len(ds)} was successfully created based on {len(files)} files')

    ds.to_csv(os.path.join(PATH, f"{DB}.csv"))
    return ds


def making_rgb_numerically(row, pbo, r0,verbose = False):
    #print(row)

    cords = eval(ds.cords.iloc[row])
    n_atoms = ds.n_atoms.iloc[row]
    atom_types = eval(ds.atom_type.iloc[row])

    s1 = time.time()
    r = distance(cords, n_atoms)
    e1 = time.time()

    s2 = time.time()
    r += ionization(atom_types, n_atoms)
    e2 = time.time()

    #g = mulliken(eval(ds.mulliken.iloc[row]), n_atoms)
    maximum_r = np.max(r)
    
    s3 = time.time()
    g = coulomb_matrix(cords, n_atoms, atom_types, diagonal = False)
    e3 = time.time()

    maximum_g = np.max(g)

    s4 = time.time()
    b = (atomic_charge(atom_types, n_atoms))
    e4 = time.time()

    s5 = time.time()
    bond_o = (bond_order(distance(cords, n_atoms), atom_types, pbo, r0))
    if sum(sum(bond_o)) == 0:
        print(row)
    b= bond_o
    e5 = time.time()

    maximum_b = np.max(b)

    if verbose == True:
        print(f'Distance matrix generation:{e1-s1}')
        print(f'Ionization matrix generation:{e2-s2}')
        print(f'Coulomb matrix generation:{e3-s3}')
        print(f'Atomic charge matrix generation:{e4-s4}')
        print(f'Bond order matrix generation:{e5-s5}')


    #making white pixels in cells in which there is no atom
    '''
    for i in range(29):
        for j in range(29):
            if i >= ds.n_atoms.iloc[row] or j>= ds.n_atoms.iloc[row]:
                b[i][j] = maximum_b
                r[i][j] = maximum_r
                g[i][j] = maximum_g
    '''


    r_max= 17.422
    r_min= 0.0
    g_max= 44.60282803468642
    g_min= 0.0
    b_max= 9.0
    b_min= 0.0



    r = (255*(r - r_min) / (r_max - r_min)).astype(int)
    g = (255*(g - g_min) / (g_max - g_min)).astype(int)
    b = (255*(b - b_min) / (b_max - b_min)).astype(int)
    return r,g,b

def calibration(ds, d, pbo, r0):
  data = []
  c = []
  start = random.randint(0,5) #to avoid too long execution
  for compound in range(start, len(ds), int(len(ds)/d)): #step != 1 to avoid too long execution

    try:
      data.append(making_rgb_numerically(compound,pbo, r0))
    except ValueError:
      print(ds.ID.iloc[compound])
      c.append(ds.ID.iloc[compound])

    if compound%10 == 0:
      print(compound)

  max_values = []
  min_values = []

  for matrices_group in zip(*data):
      max_group = np.max([np.max(matrix) for matrix in matrices_group])
      min_group = np.min([np.min(matrix) for matrix in matrices_group])
      max_values.append(max_group)
      min_values.append(min_group)

  print("r_max=", max_values[0])
  print("r_min=", min_values[0])
  print("g_max=", max_values[1])
  print("g_min=", min_values[1])
  print("b_max=", max_values[2])
  print("b_min=", min_values[2])


def making_rgb(mat, id, label):
  
  combined = np.transpose(np.array((mat[0],mat[1],mat[2])),(1,2,0))

  img = np.array(combined, dtype=np.uint8)

  pImg = Image.fromarray(img, mode='RGB')

  pImg.save(f"{PATH}/{label}/{id}.png")

def creating_images(start, end, bo, split=0.1, step=1):
    os.makedirs(f'{PATH}/{TRAIN_DIR_NAME}', exist_ok=True)
    os.makedirs(f'{PATH}/{TEST_DIR_NAME}', exist_ok = True)

    for chem in range(start, end+1, step):

        if random.randint(1,int(1/split)) == int(1/split):
            making_rgb(making_rgb_numerically(chem, pbo, r0), ds.ID.iloc[chem], label = TEST_DIR_NAME)
            print(f"{chem} goes to test set")
        else:

            making_rgb(making_rgb_numerically(chem, pbo, r0), ds.ID.iloc[chem], label=TRAIN_DIR_NAME)


    
if __name__ == '__main__':

    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--makedf', dest='making_dataframe', action='store_true', help='Creating dataframe from scratch ~15 minutes')
    parser.add_argument('--cal', dest='calibration', action='store_true', help='Calibrate numerical values of RGB representation')
    parser.add_argument('--create','-c', dest='create_files', action='store_true', help='Create images from scratch ~20 minutes')
    parser.add_argument('--start', dest='start', type = int, default = 0, help='Number of molecules to take in cosideration')
    parser.add_argument('--end', dest='end', type = int, default = 133885, help='Number of molecules to take in cosideration')
    parser.add_argument('--n', dest='quantity', type = int, default = 133885, help='Number of molecules to take in cosideration')

    args = parser.parse_args()

    start = args.start
    end = args.end
    d = args.quantity

    if args.making_dataframe:
        s = time.time()
        ds = making_df(d)
        ds[PREDICTED_VALUE] *= 27211

        
        e = time.time()
        print(f'Creating dataframe took {round(e-s,2)} seconds')
    else:
        ds = pd.read_csv(f'{DB}.csv')

    if args.calibration:
        s = time.time()
        with open('reaxff_cohnsli.lib','r') as f:
            lines = f.readlines()
            lines = [x.strip().split() for x in lines]

        pbo = lines[168:219]
        pbo = pd.DataFrame(data=pbo).drop([30,38]).reset_index(drop=True).drop([1,3],axis=1)
        pbo.columns = ['atom1','atom2','pbo1','pbo2','pbo3','pbo4','pbo5','pbo6']

        rij = lines[320:353]
        rij = pd.DataFrame(data=rij).drop([1,3,4,5,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],axis=1)
        rij.columns = ['atom1','atom2','rsigma','rpi','rpipi']


        rii = lines[42:53]
        rii = pd.DataFrame(data = rii).drop([1], axis=1)
        rii.columns = ['atom1','rsigma','rpi','rpipi']
        rii = rii.assign(atom2 = rii.atom1)

        rii = rii[['atom1','atom2','rsigma','rpi','rpipi']]

        r0 = pd.concat([rij, rii]).reset_index(drop=True)
        calibration(ds, d, pbo, r0)
        e = time.time()
        print(f"Calibration took {round(e-s,2)} seconds")

    if args.create_files:
        s = time.time()
        bo = {'CC':[-0.075,6.8316,-0.1,9.2605,-0.4591,37.7369,1.3825,1.1359,1.2104],
            'OO':[-0.1302,6.2919,-0.1239,7.6487,-0.1244,29.6439,1.2477,1.0863,0.9088],
            'CF': [-0.0953,5.7786,-0.25,15,-0.5,35,1.6,1.3144,-1],
            'OF':[-0.1,10,-0.25,15,-0.5,45,-1,-1,-1],
            'FF':[-0.1442,5.2741,-0.25,15,-0.35,25,1.162,-0.1,-1],
            'CN':[-0.115,5.8717,-0.2103,7.4487,-0.2709,29.9009,1.9263,1.4778,1.1446],
            'ON':[-0.1937,5.214,-0.2175,7.0255,-0.4462,34.9336,1.9531,1.3018,1.0984],
            'FN':[-0.1,10,-0.25,15,-0.5,45,-1,-1,-1],
            'NN':[-0.1791,5.8008,-0.205,9.7308,-0.1,19.085,1.6157,1.2558,1.0439],
            'CH':[-0.05,6.8315,0,0,0,6,1.4,1.1203,-1],
            'HH':[-0.0593,4.8358,0,0,0,6,0.7853,-0.1,-0.1],
            'CO':[-0.1463,5.2913,-0.3174,7.0303,-0.1613,10.8851,1.8523,1.2775,1.1342],
            'HO':[-0.0657,5.0451,0,0,0,6,1.68,0.9013,-1],
            'HN':[-0.0395,7.2218,0,0,0,6,2.3,1.0156,-1],
            'HF':[-0.0794,6.1403,-0.2,15,-0.2,16,-1,-1,-1],
            'FC': [-0.0953,5.7786,-0.25,15,-0.5,35,1.6,1.3144,-1],
            'FO':[-0.1,10,-0.25,15,-0.5,45,-1,-1,-1],
            'NC':[-0.115,5.8717,-0.2103,7.4487,-0.2709,29.9009,1.9263,1.4778,1.1446],
            'NO':[-0.1937,5.214,-0.2175,7.0255,-0.4462,34.9336,1.9531,1.3018,1.0984],
            'NF':[-0.1,10,-0.25,15,-0.5,45,-1,-1,-1],
            'HC':[-0.05,6.8315,0,0,0,6,1.4,1.1203,-1],
            'OC':[-0.1463,5.2913,-0.3174,7.0303,-0.1613,10.8851,1.8523,1.2775,1.1342],
            'OH':[-0.0657,5.0451,0,0,0,6,1.68,0.9013,-1],
            'NH':[-0.0395,7.2218,0,0,0,6,2.3,1.0156,-1],
            'FH':[-0.0794,6.1403,-0.2,15,-0.2,16,-1,-1,-1]}

        creating_images(start, end, bo, 0.1, 1)
        e = time.time()
        print(f"Creation of images took {round(e-s,2)} seconds")



    




