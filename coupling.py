import numpy as np
from constants import hbar, k_B
import hamiltonian
import read_files
import math_func

class coupling:
    def __init__(self, B, S, T, eigenvectors, q_vector, omega_q, masses, R_vectors, L_vectors):
        
    
        self.B = B
        self.S = S
        self.T = T
        self.Ns = int(2*self.S + 1)
        self.hdim = self.Ns
        self.eigenvectors = eigenvectors # Eigenvectors of the spin Hamiltonian

        self.masses = masses # Masses of the atoms
        self.q_vector = q_vector # Phonon mode index and wave vector q 
        self.omega_q = omega_q
        self.Nq = L_vectors.shape[0] # Number of q points
        self.Nomega = L_vectors.shape[1] # Number of phonons at q
        self.R_vectors = R_vectors # Position vectors of the cells
        self.L_vectors = L_vectors # Displacement vectors of the atoms
        self.N_atoms = len(masses)  # Number of atoms

        self.dH_dx = np.zeros((self.N_atoms,3,self.hdim,self.hdim), dtype=np.complex128)

        self.D_d1, self.G_d1,self.disp = read_files.read_d1()

        self.displacements = np.zeros((self.N_atoms, 3), dtype=np.float64)  # Initialize displacements
        self.calculate_displacements()

        self.compute_dH_dx()

        #self.dH2_dxdx = np.zeros((self.N_atoms,3,3,self.hdim,self.hdim), dtype=np.complex128)

        #self.compute_d2H_dxdx()

        self.V_alpha = np.zeros((self.Nq, self.Nomega,self.hdim, self.hdim), dtype=np.complex128)

        self.compute_V_alpha_q()

        return 
    
    def calculate_displacements(self):
        """
        Calculate displacements using the provided formula.
        
        Parameters:
        - N_q: Number of q-points
        - hbar: Reduced Planck's constant
        - omega: 2D array of frequencies [α, q]
        - masses: 1D array of particle masses [N]
        - positions: 2D array of particle positions [N, 3]
        - L: 4D array of eigenvectors [N, α, q, 3]
        - Q: 2D array of normal mode coordinates [α, q]
        - q_vectors: List of q vectors [q][3]
        
        Returns:
        - displacements: 2D array [N, 3] (complex displacements for each particle)
        """
        
        
        
        for alpha in range(self.Nomega):
            for q_idx in range(self.Nq):
                q = self.q_vector[q_idx]
                freq = self.omega_q[q_idx,alpha]
                
                
                # Bose-Einstein occupation factor
                n_q = 1 / (np.exp(freq / (k_B * self.T)) - 1)
                variance = (1/ (2 * freq)) * (1 + 2 * n_q)

              
                
                # Generate random Q_αq (complex Gaussian)
                Q_real = np.random.normal(0, np.sqrt(variance / 2))
                Q_imag = np.random.normal(0, np.sqrt(variance / 2))
                Q = Q_real + 1j * Q_imag

               

                for i in range(self.N_atoms):
                    m_i = self.masses[i]
                    R_i = self.R_vectors[i]
                    
                    q_dot_R = 2 * np.pi * np.dot(q, R_i)  # q·R in reciprocal space
                    phase = np.exp(1j * q_dot_R)
                    
                    factor = phase * Q / np.sqrt(self.Nq * m_i * freq)
                    displ = factor * self.L_vectors[q_idx,alpha,i, :]
      
                    self.displacements[i] += np.real(displ)  # Add real part of displacement

        return 
    
    def compute_dH_dx(self):
 
        # Compute the first derivative of the Hamiltonian with respect to atomic displacements  
        # second index of G_d1
        N = len(self.G_d1[0,:,0,0,0])

        dg = np.zeros((3, N, 3, 3), dtype=np.float64)
        dd = np.zeros((3, N, 3, 3), dtype=np.float64)

        for i in range(3):
            for atom in range(N):
                for j in range(3):
                    for k in range(3):

                        g_x = self.G_d1[i,atom,:,j,k]
                        d_x = self.D_d1[i,atom,:,j,k]
                        dg[i,atom,j,k] = math_func.compute_derivative(self.disp,g_x,self.displacements[atom,i])
                        dg[i,atom,j,k] = math_func.compute_derivative(self.disp,d_x,self.displacements[atom,i])

        for n in range(N):
            for i in range(3):
                dg_dxi = dg[i,n, :, :]
                dD_dxi = dd[i,n, :, :]
    
                sH = hamiltonian.hamiltonian( self.B, self.S, dg_dxi, dD_dxi)

                self.dH_dx[n,i,:,:] = sH.Hs

    def compute_V_alpha_q(self):
        """
        Compute the interaction matrix elements V^{alpha q}_{aj}.
        """
        

        mass_term = np.einsum('ik,j->ijk',self.omega_q,self.masses)
        mass_term = np.sqrt(hbar / (self.Nq * mass_term))

    #    phase_factor = np.exp(1j * np.einsum('il,kl->ik', self.q_vector, self.R_vectors))  # e^{i q . R_l}

        derivative = np.einsum('ikjl,jlab->ijkab', self.L_vectors, self.dH_dx) #dot product
        
        tmp = np.einsum('ijk,ijkab->ikab', mass_term, derivative) #Sum over atoms

        for q in range(self.Nq):
            for omega in range(self.Nomega):
                for a in range(self.hdim):
                    psi_a = self.eigenvectors[:, a]  # ⟨a|
                    for b in range(self.hdim):
                        psi_b = self.eigenvectors[:, b]  # ⟨b|
                        self.V_alpha [q,omega,a,b] = psi_a.conj().T @ tmp[q,omega,:,:] @ psi_b

        return 
    
    def compute_d2H_dxdx(self):
 
        # Compute the first derivative of the Hamiltonian with respect to atomic displacements  
        
        g_tensors=np.zeros((self.Ns, 3, 3))
        B=np.zeros(3)
        beta = np.zeros(self.Ns)
       
        for n in range(self.N_atoms):
            for m in range(self.N_atoms):
                for i in range(3):
                    for h in range(3):
                        dJ_dxi = np.zeros((self.Ns,self.Ns, 3, 3), dtype=np.complex128)
                        for s1 in range(self.Ns):    
                            for s2 in range(self.Ns):  
                                for j in range(3):
                                    for k in range(3):
                                        x = self.disp2
                                        f_x = self.J_xij[n,m,i,h,:,s1,s2,j,k]
                
                                        dJ_dxi[s1,s2,j,k] = math_func.compute_second_derivative(x,f_x)

                sH = hamiltonian.hamiltonian( B, self.S, g_tensors, beta, dJ_dxi)

                self.dH_dx[n,i,:,:] = sH.Hs
        

    




