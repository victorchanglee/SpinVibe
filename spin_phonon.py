import numpy as np
import s_hamiltonian
import coupling

""""
Run file

Set input parameters

"""


"""
Spin Hamiltonian inputs

"""

B = np.array([0.1, 0.2, 0.3])  # Magnetic field vector
g_tensors = [np.eye(3) for _ in range(3)]  # Identity g-tensors for 3 spins
spins = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]  # Spin vectors
beta = [1.0, 1.5, 2.0]  # Coupling constants

# Crystal field coefficients and operators
B_lm = [{l: {m: 0.1 for m in range(-l, l + 1)} for l in range(2, 4)} for _ in range(3)]
spin_operators = [{(l, m): 0.1 for l in range(2, 4) for m in range(-l, l + 1)} for _ in range(3)]

# Exchange coupling tensors
J_tensors = np.zeros((3, 3, 3, 3))  # Example 3x3 spin system
J_tensors[0, 1] = np.eye(3)  # Interaction between spin 0 and 1
J_tensors[1, 2] = np.eye(3)  # Interaction between spin 1 and 2

s_h = s_hamiltonian.s_hamiltonian(B, g_tensors, spins, beta, B_lm,spin_operators, J_tensors)

"""
Spin-phonon coupling inputs

"""

N_cells = 4
N_atoms = 2

# Example parameters
masses = np.array([1.0, 2.0])  # Masses of atoms
omega_alpha_q = 2.0  # Phonon frequency
x_li = np.random.rand(N_cells, 3)  # Random cell positions
L_vectors = np.random.rand(N_atoms, 3)  # Random displacement vectors
Q_alpha_q = (0, np.array([1.0, 0.0, 0.0]))  

delta = 1e-4

sp_coupling = coupling.coupling(s_h, Q_alpha_q, omega_alpha_q, masses, R_vectors, L_vectors,delta)


