import h5py
from constants import avogadro
import numpy as np

def read_orca():
    with h5py.File('Cr_otolyl_4.h5', 'r') as f:
        g_tensor = f['g_tensor'][:]
        d_tensor = f['d_tensor'][:]

    d_tensor = d_tensor
    return g_tensor, d_tensor

def read_phonons():
    with h5py.File('Sn_otolyl_4_8812.h5', 'r') as f:
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

        D_d1 = np.concatenate(([D_d1[-1]], D_d1[:-1]))
        G_d1 = np.concatenate(([G_d1[-1]], G_d1[:-1]))
        disp =np.linspace(-0.005, 0.005, 6)
        

    return D_d1, G_d1, disp

def read_atoms():
    with h5py.File('Sn_otolyl_4_atoms.h5', 'r') as f:
        R_vectors = f['positions'][:]
        masses = f['masses'][:]
        reciprocal_vectors = f['reciprocal_vectors'][:]

    masses = masses*(1E-3/avogadro) #masses in kg
    return R_vectors, masses, reciprocal_vectors

def read_indices():
    with h5py.File('isolated_indices.h5', 'r') as f:
        indices = f['indices'][:]

    return indices
