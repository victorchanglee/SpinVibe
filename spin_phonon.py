import numpy as np
import hamiltonian
import coupling
import phonon
import math_func
import redfield
import RK
import h5py as h5
from constants import eV2Ry


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
        self.S = 1  # Spin quantum number (spin-1/2)
        self.m = np.arange(-self.S, self.S+1, 1)
        self.Ns = int(2*self.S + 1)  # Number of spins
        self.dim = 2  # Dimension of the spin system
        self.hdim = self.Ns ** self.dim
        self.beta = np.random.rand(self.Ns)
        self.g_tensors = np.zeros((self.Ns, 3, 3), dtype=np.float64)  # g-tensor
        
        for i in range(self.Ns):
            self.g_tensors[i] = 2*np.eye(3)

        self.J_tensors = np.zeros((self.Ns,self.Ns, 3, 3), dtype=np.float64)
        
        for i in range(self.Ns):
            for j in range(self.Ns):
                self.J_tensors[i,j] = np.random.rand(3, 3)*20*4E-8  # Heisenberg isotropic exchange interaction coupling constant
        
        #Outputs
        self.Hs = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.eigenvalues = np.zeros(self.hdim, dtype=np.complex128)
        self.eigenvectors = np.zeros([self.hdim, self.hdim], dtype=np.complex128)

        self.Hs = self.init_s_H() #Zero displacement
        self.eigenvalues, self.eigenvectors = math_func.diagonalize(self.Hs)


        """
        Spin-phonon coupling inputs

        """
        #Inputs
        self.N_cells = 8
        self.N_atoms = 10
        self.masses = np.random.rand(self.N_atoms)*2E4  # Masses of atoms
        self.omega_q = np.array([0.05, 0.001])*eV2Ry # Phonon frequency
        self.Nq = len(self.omega_q)  # Number of phonon modes


        self.R_vectors = np.random.rand(self.N_cells, 3)*10  # Random cell positions
        self.L_vectors = np.random.rand(self.N_atoms, 3)*5  # Random displacement vectors
        self.q_vector = np.random.rand(self.Nq,3)  # Mode indices (alpha, q)
        self.disp = np.linspace(-0.0025, 0.0025, 11)
        self.N_disp = len(self.disp)

        #Outputs
        self.V_alpha_q = np.zeros([self.hdim, self.hdim, self.Nq],dtype=np.complex128)

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

        self.R1 =np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)

        self.Redfield = redfield.redfield(self.V_alpha_q,self.G_1ph)
        self.R1 = self.Redfield.R_tensor

        """
        Initialize spin density
        """

        self.m0 = 1
        self.initial_state = np.ones((self.hdim, 1), dtype=np.complex128)
        self.initial_state = self.initial_state/ np.sqrt(self.hdim)

        self.rho = np.zeros([self.hdim,self.hdim],dtype=np.complex128)

        self.rho = self.init_rho()

        """
        Perform time-evolution
        """   

        #Inputs

        self.tf = 2000
        self.dt = 1
        self.tlist = np.linspace(0, self.tf, int(self.tf / self.dt))

        #Output
        tevol = RK.RK(self.rho,self.R1,self.tf,self.tlist)

        self.drho_dt = np.zeros([self.hdim,self.hdim,len(self.tlist)],dtype=np.complex128)
        self.drho_dt = tevol.drho_dt

        """
        Save data
        """   

        self.save_data()

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
                            J_xi[n,j,i,k,l] = np.random.rand(3, 3)*20*4E-8

        V_q = coupling.coupling(self.dim, J_xi, self.S, self.q_vector, self.omega_q, self.masses, self.R_vectors, self.L_vectors, self.disp)
        self.V_alpha_q = V_q.V_alpha_q

        return

    def init_rho(self):

        if self.m0 not in self.m:
            raise ValueError(f"Invalid m0 value. Must be one of {self.m}.")


        
        # Compute density matrix ρ = |ψ⟩⟨ψ|
        rho = np.outer(self.initial_state, self.initial_state.conj())

        return rho
    
    def save_data(self):

        """
        Save data
        """

        with h5.File('data.h5', 'w') as f:
            input = f.create_group('input')
            input.create_dataset('B', data=self.B)
            input.create_dataset('tlist', data=self.tlist)

            output = f.create_group('output')
            output.create_dataset('Hs', data=self.Hs)
            output.create_dataset('drho_dt', data=self.drho_dt)

        print("Data has been saved to data.h5")

        return

spin_phonon()

 

        
