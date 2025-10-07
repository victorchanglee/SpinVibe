import spin_phonon
import numpy as np
from mpi4py import MPI
import sys
import read_files
import math_func

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Input files

orca_file = 'Cr_otolyl_4.h5'  # File containing unperturbed g and d tensors
phonon_file = 'Sn_otolyl_4_223.h5'  # File containing phonon frequencies and eigenvectors
d1_file = 'Cr_otolyl_4_d1.h5'  # File
disp1 = np.linspace(-0.05, 0.05, 10)

d2_file = 'd_predict.h5'  # File
g2_file = 'g_predict.h5'  # File
disp2 = np.array([-0.05, -0.0278, -0.0056, 0.0056, 0.0278, 0.05])

atoms_file = 'Sn_otolyl_4_atoms.h5'  # File containing atomic positions and reciprocal vectors
indices_file = 'molecule_indices.h5'  # File containing indices
mol_mass = 'mol_mass.h5'  # File containing atomic masses

if rank == 0:
        print("Input files to load:")
        print("  Unperturbed g and d tensors: ", orca_file)
        print("  Phonon frequencies and eigenvectors: ", phonon_file)
        print("  First derivatives of d and g tensors: ", d1_file)
        print("  Second derivatives of d and g tensors: ", d2_file, g2_file)
        print("  Atomic positions and reciprocal vectors: ", atoms_file)
        print("  Indices of atoms in the molecule in the crystal: ", indices_file)
        print("  Atomic masses: ", mol_mass)

file_reader = read_files.Read_files(orca_file, phonon_file, d1_file, d2_file, g2_file, atoms_file, indices_file, mol_mass, disp1, disp2)


# Input parameters

B = np.array([0.0, 0.0, 0.410]) # External magnetic field in T
S = 1 # Spin

init_type= 'pure'  # How do you initialize the spin density: polarized, pure, boltzmann, photon
R_type = None
Delta_alpha_q = float(sys.argv[2]) # Broadening parameter in cm-1
T = float(sys.argv[1]) # Temperature in Kelvin
Ncells = 50
# Rotate the molecule to match the crystal atomic coordinates
rot_mat = np.array([[0,0,1],
                    [1,0,0],
                    [0,1,0]])

#Polarization axis for initial state and magnetization calculation
pol = np.array([0,0,1])
#Rotate polzarization
axis = np.array([1, 0, 0])
theta = np.deg2rad(0)
R = math_func.rotate_polarization(axis, theta)
pol = np.dot(R, pol)

tf = 1E-3  # Total time in seconds
dt = 1E-6  # Time step in seconds

save_file = f'Spin_phonon_{T}K_{Delta_alpha_q}_{Ncells}_{theta}.h5'  # Output file to save results

spin_phonon.spin_phonon(B,S,Ncells,Delta_alpha_q,rot_mat,pol,T,tf,dt,file_reader,save_file,init_type,R_type)

if rank == 0:
        print("Job completed.")
