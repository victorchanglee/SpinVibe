import spin_phonon
import numpy as np
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

B = np.array([0.0, 0.0, 0.4]) # External magnetic field in T
S = 1 # Spin

init_type= 'polarized'  # How do you initialize the spin density
R_type = None
Delta_alpha_q = float(sys.argv[2]) # Broadening parameter in cm-1
T = float(sys.argv[1]) # Temperature in Kelvin
Ncells = 50

# Rotate the molecule to match the crystal atomic coordinates
rot_mat = np.array([[0,0,1],
                    [1,0,0],
                    [0,1,0]])

Mpol = np.array([0,0,1])

tf = 1E-3  # Total time in seconds
dt = 1E-6  # Time step in seconds

spin_phonon.spin_phonon(B,S,Ncells,Delta_alpha_q,rot_mat,Mpol,T,tf,dt,init_type,R_type)

if rank == 0:
        print("Job completed.")
