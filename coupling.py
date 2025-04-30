import numpy as np
from constants import hbar_SI, c
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
        self.indices = read_files.read_indices()
        self.N = len(self.indices) # Number of atoms in the molecule
        self.masses = masses # Masses of the atoms
        self.masses_mol = masses[self.indices] # Masses of the atoms in the molecule
        self.q_vector = q_vector # Phonon mode index and wave vector q 
        self.omega_q = omega_q
        self.Nq = L_vectors.shape[0] # Number of q points
        self.Nomega = L_vectors.shape[1] # Number of phonons at q
        self.L_vectors = L_vectors # Eigenvectors of the phonon modes
        self.R_vectors = R_vectors # Position vectors of the cells
        self.R_vectors_mol = R_vectors[self.indices,:] # Position vectors of the atoms
        self.L_vectors_mol = np.zeros((self.Nq, self.Nomega, self.N, 3), dtype=np.complex128) # Displacement vectors of the atoms
        self.L_vectors_mol = L_vectors[:,:,self.indices,:] # Displacement vectors of the atoms
        self.N_atoms = len(masses)  # Number of atoms in the crystal
        
        

        self.dH_dx = np.zeros((self.N,3,self.hdim,self.hdim), dtype=np.complex128)

        self.d_tensor, self.g_tensor = read_files.read_orca()
    

        D_d1, G_d1,disp = read_files.read_d1()
        
        self.D_d1 = D_d1
        self.G_d1 = G_d1
        self.disp = disp


        #self.displacements = np.zeros((self.N, 3), dtype=np.float64)  # Initialize displacements

        #self.calculate_displacements()

        self.compute_dH_dx()

        #self.dH2_dxdx = np.zeros((self.N_atoms,3,3,self.hdim,self.hdim), dtype=np.complex128)

        #self.compute_d2H_dxdx()

        self.V_alpha = np.zeros((self.Nq, self.Nomega,self.hdim, self.hdim), dtype=np.complex128)

        self.compute_V_alpha_q()

        return 

    
    def compute_dH_dx(self):
 
        # Compute the first derivative of the Hamiltonian with respect to atomic displacements  
        # second index of G_d1
        

        dg = np.zeros((3, self.N, 3, 3), dtype=np.float64)
        dd = np.zeros((3,  self.N, 3, 3), dtype=np.float64)

        for i in range(3):
            for atom in range(self.N):
                for j in range(3):
                    for k in range(3):

                        g_x = self.G_d1[i,atom,:,j,k]
                        d_x = self.D_d1[i,atom,:,j,k]
                        dg[i,atom,j,k] = math_func.compute_derivative(self.disp, g_x)
                        dd[i,atom,j,k] = math_func.compute_derivative(self.disp, d_x)


        for n in range( self.N):
            for i in range(3):
                dg_dxi = dg[i,n, :, :]
                dD_dxi = dd[i,n, :, :]
    
                sH = hamiltonian.hamiltonian( self.B, self.S, dg_dxi, dD_dxi)

                self.dH_dx[n,i,:,:] = sH.Hs

    def compute_V_alpha_q(self):
        """
        Compute the interaction matrix elements V^{alpha q}_{aj}.
        """

        tmp1 = np.zeros((self.Nq, self.Nomega,self.N), dtype=np.complex128)
        term = np.zeros((self.Nq, self.Nomega,self.N, self.hdim,self.hdim), dtype=np.complex128)
        tmp = np.zeros((self.Nq, self.Nomega,self.hdim,self.hdim), dtype=np.complex128)

        for q in range(self.Nq):
            for omega in range(self.Nomega):
                for atoms in range(self.N):
                    freq = 2*np.pi * self.omega_q[q,omega] *c  # Frequency in rad/s
                    # Compute the prefactor
                    prefactor = np.sqrt(hbar_SI / (self.N*self.Nq*self.Nomega *freq * self.masses_mol[atoms])) * 1E10  # Convert to Angstroms
                    
                    # Compute the exponential term (dot product of q1 and R_l)
                    q_dot_R = np.dot(self.q_vector[q,:], self.R_vectors_mol[atoms,:])
                    
                    exponential = np.exp(1j * q_dot_R)  # Complex exponential
                    
                    tmp1[q,omega,atoms] = prefactor*exponential

                    
                    # Compute the full term

        term = np.einsum('ijkl,klab ->ijkab', self.L_vectors_mol, self.dH_dx) #dot product

        tmp = np.einsum('ijk, ijkab -> ijab', tmp1, term) #sum over atoms
        
        #Impose Hermiticity
        
        H_herm = np.conj(np.transpose(tmp, axes=(0, 1, 3, 2)))
        H_symmetrized = 0.5 * (tmp + H_herm)


        self.V_alpha = H_symmetrized
        
        return 


    




