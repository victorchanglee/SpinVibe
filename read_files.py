import h5py
from constants import avogadro
import numpy as np
import math_func

def read_orca():
    with h5py.File('Cr_otolyl_4.h5', 'r') as f:
        g_tensor = f['g_tensor'][:]
        d_tensor = f['d_tensor'][:]

    return g_tensor, d_tensor

def read_phonons():
    with h5py.File('Sn_otolyl_4_223.h5', 'r') as f:
        q_points = f['q_points'][1:, :]
        frequencies_cm = f['frequencies_cm'][1:, :]
        eigenvectors = f['eigenvectors'][1:, :]

    omega_q = frequencies_cm
    eigenvectors = eigenvectors
            
    return q_points, omega_q, eigenvectors

def read_d1():
    with h5py.File('Cr_otolyl_4_d1.h5', 'r') as f:
        D_d1 = f['d_tensor'][:]
        G_d1 = f['g_matrix'][:]
    
    disp1 =np.linspace(-0.05, 0.05, 10)
        

    return D_d1, G_d1, disp1

def read_d2():

    disp = np.array([-0.05, -0.0278, -0.0056, 0.0056, 0.0278, 0.05])
    disp2 = np.stack([disp, disp], axis=0)


    with h5py.File('d_predict.h5', 'r') as f:
        D_d2 = f['d_tensor'][:]

    with h5py.File('g_predict.h5', 'r') as f:
        G_d2 = f['g_tensor'][:]
 
    return D_d2, G_d2, disp2

def read_atoms():
    with h5py.File('Sn_otolyl_4_atoms.h5', 'r') as f:
        R_vectors = f['positions'][:]
        reciprocal_vectors = f['reciprocal_vectors'][:]

    return R_vectors, reciprocal_vectors

def read_indices():
    with h5py.File('molecule_indices.h5', 'r') as f:
        indices = f['indices'][:]

    return indices

def read_mol_masses():
    with h5py.File('mol_mass.h5', 'r') as f:
        masses = f['atomic_masses'][:]

    masses = masses*(1E-3/avogadro) #masses in kg
    masses = np.concatenate(([masses[-1]], masses[:-1]))
    return masses