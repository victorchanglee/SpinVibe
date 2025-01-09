import numpy as np
import s_hamiltonian
import coupling
import phonon

""""
Run file

Set input parameters

"""


"""
Spin Hamiltonian inputs

"""
def init_s_H():

    B = np.array([0.1, 0.2, 0.3])  # Magnetic field vector
    g_tensors = [np.eye(3) for _ in range(3)]  # Identity g-tensors for 3 spins
    spins = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]  # Spin vectors
    beta = np.random.rand(3) # Random Coupling constants

    # Crystal field coefficients and operators
    B_lm = [{l: {m: 0.1 for m in range(-l, l + 1)} for l in range(2, 4)} for _ in range(3)]
    spin_operators = [{(l, m): 0.1 for l in range(2, 4) for m in range(-l, l + 1)} for _ in range(3)]

    # Exchange coupling tensors
    D_tensors = np.zeros((3, 3, 3, 3))  # trace-less Cartesian tensor - Exchange anysotropy
    D_tensors[0, 1] = np.eye(3)  
    D_tensors[1, 2] = np.eye(3)  

    J_tensors = np.zeros((3, 3, 3, 3))  # Heisenberg isotropic exchange interaction coupling constant
    J_tensors[0, 1] = np.eye(3) 
    J_tensors[1, 2] = np.eye(3)  
    
    s_H = s_hamiltonian.s_hamiltonian(B, g_tensors, spins, beta, B_lm,spin_operators, D_tensors, J_tensors)

    return s_H.H_s

"""
Spin-phonon coupling inputs

"""

N_cells = 8
N_atoms = 10
N_q = 1

masses = np.random.rand(N_atoms)  # Masses of atoms
omega_alpha_q = 2.0  # Phonon frequency
R_vectors = np.random.rand(N_cells, 3)  # Random cell positions
L_vectors = np.random.rand(N_atoms, 3)  # Random displacement vectors
q_vector = np.array([0.1, 0.2, 0.3])  # Mode indices (alpha, q)

#x_li = np.random.rand(N_cells, N_atoms, 3)  # Random atomic displacements

disp = np.linspace(-0.0025, 0.0025, 11)
N_disp = len(disp)

Hs_xi = np.zeros([N_atoms,3,N_disp], dtype=np.float64)

for n in range(N_atoms):
    for j in range(3):
        for i in range(N_disp):
            Hs_xi[n,j,i] = init_s_H()



sp_coupling = coupling.coupling(Hs_xi, N_q, q_vector, omega_alpha_q, masses, R_vectors, L_vectors,disp)

G_1ph = phonon.phonon(1.0, omega_alpha_q, Delta_alpha_q, n_alpha_q)
