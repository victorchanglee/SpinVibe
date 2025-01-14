import numpy as np
from constants import hbar
import math_func
import hamiltonian

class coupling:
    def __init__(self, dim, J_xi, S, q_vector, omega_q, masses, R_vectors, L_vectors, disp):
        
        self.dim = dim 
        self.J_xi = J_xi   # Exchange interaction
        self.S = S
        self.Ns = int(2*self.S + 1)
        self.hdim = self.Ns ** self.dim    
    
        self.q_vector = q_vector # Phonon mode index and wave vector q 
        self.omega_q = omega_q  # Phonon frequencies
        self.Nq = len(self.omega_q) # Number of phonon modes
        self.masses = masses  # Masses of the atoms
        self.R_vectors = R_vectors # Position vectors of the cells
        self.L_vectors = L_vectors # Displacement vectors of the atoms
        self.N_cells = len(R_vectors)  # Number of cells
        self.N_atoms = len(L_vectors)  # Number of atoms
        self.hbar = hbar # Planck's constant
        self.disp = disp  # Displacement values
        self.N_disp = len(disp)  # Number of displacement values

        self.dH_dx = np.zeros((self.N_atoms,3,self.hdim,self.hdim), dtype=np.complex128)

        self.compute_dH_dx()

        self.V_alpha_q = np.zeros((self.hdim, self.hdim, self.Nq), dtype=np.complex128)

        self.compute_V_alpha_q()

        return 
    
    def compute_dH_dx(self):
 
        # Compute the first derivative of the Hamiltonian with respect to atomic displacements  
        
        g_tensors=np.zeros((self.Ns, 3, 3))
        B=np.zeros(3)
        beta = np.zeros(self.Ns)
       
        for n in range(self.N_atoms):
            for i in range(3):

                dJ_dxi = np.zeros((self.Ns,self.Ns, 3, 3), dtype=np.complex128)
    
                for s1 in range(self.Ns):    
                    for s2 in range(self.Ns):  
                        for j in range(3):
                            for k in range(3):
                                x = self.disp
                                f_x = self.J_xi[n,i,:,s1,s2,j,k]
        
                                dJ_dxi[s1,s2,j,k] = math_func.compute_derivative(x,f_x)

                sH = hamiltonian.hamiltonian( B, self.S, self.dim, g_tensors, beta, dJ_dxi)

                self.dH_dx[n,i,:,:] = sH.Hs

    def compute_V_alpha_q(self):
        """
        Compute the interaction matrix elements V^{alpha q}_{aj}.
        """

        mass_term = np.einsum('i,j->ij',self.omega_q,self.masses)
        mass_term = np.sqrt(self.hbar / (self.Nq * mass_term))
        
        phase_factor = np.exp(1j * np.einsum('ij,kj->ik', self.q_vector, self.R_vectors))  # e^{i q . R_l}

        tmp1 = np.einsum('ij,ijkl->ikl', self.L_vectors, self.dH_dx) #Sum over cartesian
        tmp2 = np.einsum('ij,jkl->ikl', mass_term, tmp1) #Sum over atoms
        tmp3 = np.einsum('ij,ikl->kli', phase_factor, tmp2) #Sum over cells

        self.V_alpha_q = tmp3 

        return
    
    




