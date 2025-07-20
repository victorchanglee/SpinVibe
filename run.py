import spin_phonon
import numpy as np
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

B = np.array([0, 0, 0])
S = 1

init_type= 'polarized'  # 'polarized', 'boltzmann'
Delta_alpha_q = 1 # Broadening parameter
T = float(sys.argv[1]) # Temperature in Kelvin

tf = 1E-4  # Total time in seconds
dt = 1E-10  # Time step in seconds

spin_phonon.spin_phonon(B,S,Delta_alpha_q,T,tf,dt,init_type)

if rank == 0:
        print("Job completed.")