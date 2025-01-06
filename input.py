import numpy as np
import spin_phonon

""""
Run file

Set input parameters

"""

B = np.array([0.1, 0.2, 0.3])  # Magnetic field vector
g_tensors = [np.eye(3) for _ in range(3)]  # Identity g-tensors for 3 spins
spins = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]  # Spin vectors
beta = [1.0, 1.5, 2.0]  # Coupling constants
D_tensor = np.zeros((3, 3, 3, 3))  # Spin-spin interaction tensor
D_tensor[0, 1] = np.eye(3)  # Example interaction between spin 0 and 1
D_tensor[1, 0] = np.eye(3)  # Symmetric interaction

spin_phonon = spin_phonon.spin_phonon(B, g_tensors, spins, beta, D_tensor)

print(spin_phonon.H_s)
