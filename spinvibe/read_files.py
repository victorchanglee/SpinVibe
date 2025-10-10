import h5py
from .constants import avogadro
import numpy as np
from . import math_func

class Read_files:
    def __init__(self, spin_file, phonon_file, d1_file, d2_file, g2_file, atoms_file, indices_file, mol_mass, disp1, disp2):
        self.spin_file = spin_file
        self.phonon_file = phonon_file
        self.d1_file = d1_file
        self.d2_file = d2_file
        self.g2_file = g2_file
        self.atoms_file = atoms_file
        self.indices_file = indices_file
        self.mol_mass = mol_mass
        self.disp1 = disp1
        self.disp2 = disp2

    def read_spin(self):
        with h5py.File(self.spin_file, 'r') as f:
            g_tensor = f['g_tensor'][:]
            d_tensor = f['d_tensor'][:]

        return g_tensor, d_tensor

    def read_phonons(self):

        with h5py.File(self.phonon_file, 'r') as f:
            q_points = f['q_points'][1:, :]
            frequencies_cm = f['frequencies_cm'][1:, :]
            eigenvectors = f['eigenvectors'][1:, :]

        omega_q = frequencies_cm
        eigenvectors = eigenvectors
                
        return q_points, omega_q, eigenvectors

    def read_d1(self):

        with h5py.File(self.d1_file, 'r') as f:
            D_d1 = f['d_tensor'][:]
            G_d1 = f['g_matrix'][:]
        
        disp1 = self.disp1

        return D_d1, G_d1, disp1

    def read_d2(self):


        disp = self.disp2
        disp2 = np.stack([disp, disp], axis=0)


        with h5py.File(self.d2_file, 'r') as f:
            D_d2 = f['d_tensor'][:]

        with h5py.File(self.g2_file, 'r') as f:
            G_d2 = f['g_tensor'][:]
    
        return D_d2, G_d2, disp2

    def read_atoms(self):


        with h5py.File(self.atoms_file, 'r') as f:
            R_vectors = f['positions'][:]
            reciprocal_vectors = f['reciprocal_vectors'][:]

        return R_vectors, reciprocal_vectors

    def read_indices(self):


        with h5py.File(self.indices_file, 'r') as f:
            indices = f['indices'][:]

        return indices

    def read_mol_masses(self):


        with h5py.File(self.mol_mass, 'r') as f:
            masses = f['atomic_masses'][:]

        masses = masses*(1E-3/avogadro) #masses in kg
        masses = np.concatenate(([masses[-1]], masses[:-1]))
        return masses
