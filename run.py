import spin_phonon
import argparse
import os
import numpy as np

if __name__ == "__main__":
    

    B = np.array([0, 0, 0.333])
    S = 1

    init_type= 'polarized'  
    Delta_alpha_q = 0.001  # Broadening parameter
    T = 10

    tf = 100  # Total time
    dt = 1E-3  # Time step



    spin_phonon.spin_phonon(B,S,Delta_alpha_q,T,tf,dt,init_type)
