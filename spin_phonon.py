import numpy as np
import s_hamiltonian
import coupling
import phonon
import math_func
""""
Run file

Set input parameters

"""
class spin_phonon:
    def __init__(self):
        
        """
        Spin Hamiltonian inputs

        """
        
        self.B = np.array([0.1, 0.2, 0.3])  # Magnetic field vector
        
        self.S = 1/2  # Spin quantum number (spin-1/2)
        self.Ns = int(2*self.S + 1)  # Number of spins
        self.dim = 2  # Dimension of the spin system
        self.h_dim = self.Ns ** self.dim
        self.g_tensors = np.zeros((self.Ns, 3, 3), dtype=np.float64)  # g-tensor

        for i in range(self.Ns):
            self.g_tensors[i] = 2*np.eye(3)

        self.J_tensors = np.zeros((self.Ns,self.Ns, 3, 3), dtype=np.float64)
        
        for i in range(self.Ns):
            for j in range(self.Ns):
                self.J_tensors[i,j] = np.random.rand(3, 3)  # Heisenberg isotropic exchange interaction coupling constant
        
        

        self.beta = np.random.rand(self.Ns)


        self.H_s = np.zeros([self.h_dim, self.h_dim], dtype=np.complex128)
        self.eigenvalues = np.zeros(self.h_dim, dtype=np.complex128)
        self.eigenvectors = np.zeros([self.h_dim, self.h_dim], dtype=np.complex128)

        self.H_s = self.init_s_H() #Zero displacement

        self.eigenvalues, self.eigenvectors = math_func.diagonalize(self.H_s)



        """
        Spin-phonon coupling inputs

        """
        self.N_cells = 8
        self.N_atoms = 10
        self.N_q = 1

        self.masses = np.random.rand(self.N_atoms)  # Masses of atoms
        self.omega_alpha_q = 2.0  # Phonon frequency
        self.R_vectors = np.random.rand(self.N_cells, 3)  # Random cell positions
        self.L_vectors = np.random.rand(self.N_atoms, 3)  # Random displacement vectors
        self.q_vector = np.array([0.1, 0.2, 0.3])  # Mode indices (alpha, q)

        #x_li = np.random.rand(N_cells, N_atoms, 3)  # Random atomic displacements

        self.disp = np.linspace(-0.0025, 0.0025, 11)
        self.N_disp = len(self.disp)


        self.sp_coupling = None
        
        self.Hs_xi = np.zeros([self.N_atoms,3,self.N_disp,self.h_dim], dtype=np.complex128)

        self.init_sp_coupling()

        return


    def init_s_H(self):



         # Random Coupling constants
        
        # Crystal field coefficients and operators
        #B_lm = [{l: {m: 0.1 for m in range(-l, l + 1)} for l in range(2, 4)} for _ in range(3)]
        

        # Exchange coupling tensors
        #D_tensors = np.zeros((3, 3, 3, 3))  # trace-less Cartesian tensor - Exchange anysotropy
        #D_tensors[0, 1] = np.eye(3)  
        #D_tensors[1, 2] = np.eye(3)  

        
        s_H = s_hamiltonian.s_hamiltonian(self.B, self.S, self.dim, self.g_tensors, self.beta, self.J_tensors)
        
        return s_H.H_s

    

    def init_sp_coupling(self):

        for n in range(self.N_atoms):
            for j in range(3):
                for i in range(self.N_disp):


                    J_tensor = np.zeros((self.Ns,self.Ns, 3, 3), dtype=np.float64)
        
                    for k in range(self.Ns):
                        for l in range(self.Ns):
                            J_tensor[k,l] = np.random.rand(3, 3)
     
                    Hs_xi = s_hamiltonian.s_hamiltonian(self.B, self.S, self.dim, self.g_tensors, self.beta, J_tensor)
                    eigenvalues, eigenvectors = math_func.diagonalize(Hs_xi.H_s)
                    self.Hs_xi[n,j,i,:] = eigenvalues

        
        

        self.sp_coupling = coupling.coupling(self.Hs_xi, self.N_q, self.q_vector, self.omega_alpha_q, self.masses, self.R_vectors, self.L_vectors, self.disp)

        return
    





#        G_1ph = phonon.phonon(1.0, omega_alpha_q, Delta_alpha_q, n_alpha_q)
