import numpy as np
import pandas as pd
from ase.data import atomic_numbers
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.params import *

import logging
from scripts.logging import setup_logging
setup_logging()
#matrices that take values not only on the diagonal

#distance matrix

def distance(coordinates, n_atoms):
    dist_mat = np.zeros((n_atoms,n_atoms))

    min_length = min(n_atoms, len(coordinates))

    for i in range(min_length):
        for j in range(min_length):
            
            if i != j:
                try:
                    cord_i = np.array(np.char.replace(coordinates[i], '*^', 'e')).astype(float)
                    cord_j = np.array(np.char.replace(coordinates[j], '*^', 'e')).astype(float)
                except ValueError:
                    logging.DEBUG(f"There is problem with {coordinates} set of values")
                
                dist_mat[i][j] = np.linalg.norm(cord_i - cord_j)
    return dist_mat

def bond_order(mat, atom_type, bo):
    n = len(atom_type)
    for i in range(n):
        for j in range(n):
            if i!=j:
                pair = f'{atom_type[i]}{atom_type[j]}'

                pbo1, pbo2, pbo3, pbo4, pbo5, pbo6, rsigma, rpi, rpipi = bo[pair]

                BO_sigma = np.exp(pbo1*(mat[i][j]/rsigma)**pbo2) if rsigma != float(-1) else 0
                BO_pi = np.exp(pbo3*(mat[i][j]/rpi)**pbo4) if rpi != float(-1) else 0
                BO_pipi = np.exp(pbo5*(mat[i][j]/rpipi)**pbo6) if rpipi != float(-1) else 0

                mat[i][j] = BO_sigma+BO_pi+BO_pipi
                if mat[i][j] >5: #to track bugs and unusual cases
                    print(i,j, atom_type[i], atom_type[j], BO_pi, rpi, pbo3, pbo4)

    return mat

def coulomb_matrix(coordinates, n_atoms, atom_type, diagonal = True):
    
    c_mat = np.zeros((n_atoms, n_atoms))
    Z = atomic_numbers
    min_len = min(n_atoms, len(coordinates))

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i==j:
                c_mat[i][i] = 0.5*Z[atom_type[i]]**2.4 if diagonal == True else 0

            else:
                cord_i = np.array(np.char.replace(coordinates[i], '*^', 'e')).astype(float)
                cord_j = np.array(np.char.replace(coordinates[j], '*^', 'e')).astype(float)
                c_mat[i][j] = Z[atom_type[i]]*Z[atom_type[j]] / np.linalg.norm(cord_i-cord_j)
    return c_mat

#matrices that take values only on the diagonal

#making mulliken matrix
def mulliken(m, n_atoms):
    m_mat = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        m_mat[i][i] = float(m[i].replace('*^','e'))

    return m_mat

#making atomic charge matrix
def atomic_charge(atom_type, n_atoms):
    a_mat = np.zeros((n_atoms, n_atoms))

    atom_dict = atomic_numbers
    for i in range(n_atoms):
        a_mat[i][i] = atom_dict[atom_type[i]]

    return a_mat


#making electronegativity matrix
def electronegativity(atom_type, n_atoms):
    a_mat = np.zeros((n_atoms, n_atoms))


    atom_dict = {'H':2.20, 'O':3.44, 'C':2.55, 'N':3.04, 'F':3.98}
    for i in range(n_atoms):
        a_mat[i][i] = atom_dict[atom_type[i]]

    return a_mat

#making electron affinity matrix
def electronaffinity(atom_type, n_atoms):
    a_mat = np.zeros((n_atoms, n_atoms))


    atom_dict = {'H':0.755, 'O':1.46, 'C':1.595, 'N':0.07, 'F':3.40}
    for i in range(n_atoms):
        a_mat[i][i] = atom_dict[atom_type[i]]

    return a_mat

def ionization(atom_type, n_atoms):
    a_mat = np.zeros((n_atoms, n_atoms))
    
    atom_dict = atomic_numbers
    for i in range(n_atoms):
        try:
            a_mat[i][i] = atom_dict[atom_type[i]]
        except KeyError:
            logging.DEBUG(f"{atom_type[i]} is not present in dictionary, put 0 instead of appriopiate ionization energy")
            
            
    return a_mat


   
