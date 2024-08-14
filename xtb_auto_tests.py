import os 
import sys
import pandas as pd

df = pd.read_csv('qm7_vanilla.csv')

l = df.ID.to_list()
data_list = [f"dsgdb9nsd_{'0'*(6-len(x.split('_')[1]))}{x.split('_')[1]}.xyz" for x in l]

#data_list = os.listdir('./data')

def xyz_to_xtb(filename):

    #read file
    with open(filename, 'r') as f:
        lines = f.readlines()

    try:
        #save info
        n_atoms = int(lines[0])
        properties = lines[1]
        cords_raw = lines[2:2+n_atoms]

        #edit cords info
        cords_edited = []
        for atom in cords_raw:
            a = atom.split()
            atom_edited = []
            for x in a[:-1]:
                try:
                    atom_edited.append(float(x))
                except ValueError:
                    atom_edited.append(x)
            cords_edited.append(atom_edited)
        

        input_name = filename.replace('.xyz','xtb.xyz')
        #print cords info in the form of xtb input

        with open(input_name, 'w') as input:

            input.write(f"{n_atoms}\n")
            input.write('\n')

            for line in cords_edited:
                input.write(f"{line[0]} {line[1]} {line[2]} {line[3]} \n")
        try:
            os.system(f"/sw/xtb-dist/bin/xtb {input_name} --opt extreme> tmp.txt")
            

            with open('tmp.txt', 'r') as f:
                output_lines = f.readlines()

                bandgap = [x for x in output_lines if 'HOMO-LUMO GAP' in x][0]
                energy = [x for x in output_lines if 'TOTAL ENERGY' in x][0]
            
            bandgap = bandgap.split('.')
            energy = energy.split('.')

            actual_bandgap = float(f"{bandgap[0][-4:]}.{bandgap[1][:8]}")
            actual_energy = float(f"{energy[0][-4:]}.{energy[1][:8]}")
            print(f'\rCalculated bandgap: {actual_bandgap}, energy: {actual_energy}',end='')

            return actual_bandgap, actual_energy
        except:
            print('Something went wrong with xtb calculations')
    except:
        print('Something went wrong with input preparation')

    return 1000, 1000


if __name__ == '__main__':
    bandgap = []
    energy = []
    for i,file in enumerate(data_list):
        print(f"\r{i}/{len(data_list):.2f}",end='')
        res = xyz_to_xtb(f"data/{file}")
        bandgap.append(res[0])
        energy.append(res[1])

    
    df['xtb_bandgap'] = bandgap
    df['xtb_energy'] = energy


    df.to_csv('testowanko.csv')
        
