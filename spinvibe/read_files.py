import h5py
from .constants import avogadro
import numpy as np
from . import math_func

class Read_files:
    def __init__(self, spin_file, phonon_file, derivatives_file, atoms_file, indices_file):
        self.spin_file = spin_file
        self.phonon_file = phonon_file
        self.derivative_file = derivatives_file
        self.atoms_file = atoms_file
        self.indices_file = indices_file


    def read_spin(self):
        with h5py.File(self.spin_file, 'r') as f:
            g_tensor = f['g_tensor'][:]
            d_tensor = f['d_tensor'][:]

        return g_tensor, d_tensor

    def read_phonons(self):

        with h5py.File(self.phonon_file, 'r') as f:
            q_points = f['q_points'][:, :]
            frequencies_cm = f['frequencies_cm'][:, :]
            eigenvectors = f['eigenvectors'][:, :]

        omega_q = frequencies_cm
        eigenvectors = eigenvectors
                
        return q_points, omega_q, eigenvectors

    def read_derivatives(self):

        with h5py.File(self.derivative_file, 'r') as f:
            D_d1 = f['d1'][:]
            G_d1 = f['g1'][:]
            D_d2 = f['d2'][:]
            G_d2 = f['g2'][:]


        return D_d1, G_d1, D_d2, G_d2


    def read_atoms(self):


        with h5py.File(self.atoms_file, 'r') as f:
            R_vectors = f['lattice_vectors'][:]
            reciprocal_vectors = f['reciprocal_vectors'][:]
            masses = f['masses'][:]

        masses = masses*(1E-3/avogadro) #masses in kg
            
        return R_vectors, reciprocal_vectors, masses

    def read_indices(self):


        with h5py.File(self.indices_file, 'r') as f:
            indices = f['indices'][:]

        return indices

