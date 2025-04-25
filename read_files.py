import h5py
from constants import avogadro,kg2ev,eV2cm
import numpy as np

def read_orca():
    with h5py.File('Cr_otolyl_4.h5', 'r') as f:
        g_tensor = f['g_tensor'][:]
        d_tensor = f['d_tensor'][:]

    d_tensor = d_tensor/eV2cm
    return g_tensor, d_tensor

def read_phonons():
    with h5py.File('Sn_otolyl_4_phonon.h5', 'r') as f:
        q_points = f['q_points'][1:, :]
        frequencies_cm = f['frequencies_cm'][1:, :]
        eigenvectors = f['eigenvectors'][1:, :]

    omega_q = frequencies_cm/eV2cm
            
    return q_points, omega_q, eigenvectors

def read_d1():
    with h5py.File('Cr_otolyl_4_d1.h5', 'r') as f:
        D_d1 = f['d_tensor'][:]
        G_d1 = f['g_matrix'][:]

        D_d1 = D_d1/eV2cm
        disp =np.linspace(-0.005, 0.005, 6)

    return D_d1, G_d1, disp

def read_atoms():
    with h5py.File('Sn_otolyl_4_atoms.h5', 'r') as f:
        R_vectors = f['positions'][:]
        masses = f['masses'][:]
        reciprocal_vectors = f['reciprocal_vectors'][:]

    masses = masses*(1E-3/avogadro)*kg2ev
    return R_vectors, masses, reciprocal_vectors