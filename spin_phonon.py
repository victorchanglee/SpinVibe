import numpy as np
import hamiltonian
import coupling
import phonon
import math_func
import redfield

""""
Run file

Set input parameters

"""
class spin_phonon:
    def __init__(self):
        
        """
        Spin Hamiltonian inputs

        To Do: Crystal field

        """
        #Inputs
        self.B = np.array([0.1, 0.2, 0.3])  # Magnetic field vector
        self.S = 1/2  # Spin quantum number (spin-1/2)
        self.Ns = int(2*self.S + 1)  # Number of spins
        self.dim = 2  # Dimension of the spin system
        self.hdim = self.Ns ** self.dim
        self.g_tensors = np.zeros((self.Ns, 3, 3), dtype=np.float64)  # g-tensor
        self.beta = np.random.rand(self.Ns)

        for i in range(self.Ns):
            self.g_tensors[i] = 2*np.eye(3)

        self.J_tensors = np.zeros((self.Ns,self.Ns, 3, 3), dtype=np.float64)
        
        for i in range(self.Ns):
            for j in range(self.Ns):
                self.J_tensors[i,j] = np.random.rand(3, 3)  # Heisenberg isotropic exchange interaction coupling constant
        
        
        #Outputs
        self.H_s = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.eigenvalues = np.zeros(self.hdim, dtype=np.complex128)
        self.eigenvectors = np.zeros([self.hdim, self.hdim], dtype=np.complex128)

        self.H_s = self.init_s_H() #Zero displacement
        self.eigenvalues, self.eigenvectors = math_func.diagonalize(self.H_s)

        

        """
        Spin-phonon coupling inputs

        """
        #Inputs
        self.N_cells = 8
        self.N_atoms = 10
        self.masses = np.random.rand(self.N_atoms)  # Masses of atoms
        self.omega_q = np.array([1.0, 2.0]) # Phonon frequency
        self.Nq = len(self.omega_q)  # Number of phonon modes
        self.R_vectors = np.random.rand(self.N_cells, 3)  # Random cell positions
        self.L_vectors = np.random.rand(self.N_atoms, 3)  # Random displacement vectors
        self.q_vector = np.random.rand(self.Nq,3)  # Mode indices (alpha, q)
        self.disp = np.linspace(-0.0025, 0.0025, 11)
        self.N_disp = len(self.disp)

        #Outputs
        self.V_alpha_q = np.random.rand(self.hdim, self.hdim, self.Nq)

        self.init_sp_coupling()


        """
        Green's function inputs
        """        
        #Inputs
        self.omega_ij = np.zeros([self.hdim,self.hdim], dtype=np.float64)
        self.omega_ij = math_func.energy_diff(self.eigenvalues)
        self.Delta_alpha_q = 0.1  # Broadening parameter
        self.T = 300  # Temperature in Kelvin

        #Outputs
        self.G_1ph = np.zeros([self.hdim,self.hdim,self.Nq], dtype=np.float64)
        self.G_ph = phonon.phonon(self.omega_ij, self.omega_q, self.Delta_alpha_q,self.T)
        self.G_1ph = self.G_ph.G_1ph_value

        """
        Redfield equation
        """      

        self.R_tensor =np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)

        self.Redfield = redfield.redfield(self.V_alpha_q,self.G_1ph)
        self.R_tensor = self.Redfield.R_tensor

        

        return


    def init_s_H(self):
     
        sH = hamiltonian.hamiltonian(self.B, self.S, self.dim, self.g_tensors, self.beta, self.J_tensors)
        
        return sH.Hs

    

    def init_sp_coupling(self):

        """
        Compute the spin-phonon coupling energy.

        Parameters:
   
        Returns:    
        """
        
        J_xi = np.zeros([self.N_atoms,3,self.N_disp,self.Ns,self.Ns, 3, 3], dtype=np.complex128)
       
  
        for n in range(self.N_atoms):
            for j in range(3):
                for i in range(self.N_disp):
                    for k in range(self.Ns):
                        for l in range(self.Ns):
                            J_xi[n,j,i,k,l] = np.random.rand(3, 3)

        V_q = coupling.coupling(self.dim, J_xi, self.S, self.q_vector, self.omega_q, self.masses, self.R_vectors, self.L_vectors, self.disp)
        self.V_alpha_q = V_q.V_alpha_q

        return





 

        
