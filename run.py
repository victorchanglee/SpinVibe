import spin_phonon
import numpy as np

if __name__ == "__main__":
    

    B = np.array([0, 0, 0.41])
    S = 1

    init_type= 'polarized'  # 'polarized', 'boltzmann'
    Delta_alpha_q = 1 # Broadening parameter
    T = 40 # Temperature in Kelvin

    tf = 1E-2  # Total time in seconds
    dt = 1E-6  # Time step in seconds

    spin_phonon.spin_phonon(B,S,Delta_alpha_q,T,tf,dt,init_type)
