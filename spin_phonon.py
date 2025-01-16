import numpy as np
import hamiltonian
import coupling
import phonon
import math_func
import redfield
import RK
import measure
import h5py as h5
from constants import mass
import itertools

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
        self.B = np.array([0.0, 0.0, 7])  # Magnetic field vector
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
                self.J_tensors[i,j] = np.random.rand(3, 3)*20*4E-4  # Heisenberg isotropic exchange interaction coupling constant
        
        #Outputs
        self.Hs = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.eigenvalues = np.zeros(self.hdim, dtype=np.complex128)
        self.eigenvectors = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.S_operator = np.zeros((self.Ns, self.Ns,3),dtype=np.complex128)


        self.Hs = self.init_s_H() #Zero displacement
        self.eigenvalues, self.eigenvectors = math_func.diagonalize(self.Hs)


        """
        Spin-phonon coupling inputs

        """
        #Inputs
        self.N_cells = 8
        self.N_atoms = 10
        self.masses = np.random.rand(self.N_atoms)*20*mass  # Masses of atoms
        self.omega_q = np.array([0.05, 0.001]) # Phonon frequency
        self.Nq = len(self.omega_q)  # Number of phonon modes




        #Outputs
        self.V_alpha_q = np.zeros([self.hdim, self.hdim, self.Nq],dtype=np.complex128)
        self.V_alpha_beta = np.zeros([self.hdim, self.hdim, self.Nq],dtype=np.complex128)
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
        self.G_2ph = np.zeros([self.hdim,self.hdim,self.Nq,self.Nq], dtype=np.float64)
        self.G_ph = phonon.phonon(self.omega_ij, self.omega_q, self.Delta_alpha_q,self.T)
        self.G_1ph = self.G_ph.G_1ph_value
        self.G_2ph = self.G_ph.G_2ph_value

        """
        Redfield equation
        """      

        self.R1 =np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
        self.R2 =np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)

        self.Redfield = redfield.redfield(self.V_alpha_q,self.V_alpha_beta,self.G_1ph,self.G_2ph)
        self.R1 = self.Redfield.R1_tensor
        self.R2 = self.Redfield.R2_tensor

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

        self.tf = 1E-4
        self.dt = 1E-8
        self.tlist = np.linspace(0, self.tf, int(self.tf / self.dt))
        self.tsteps = len(self.tlist)
        #Output
        tevol = RK.RK(self.rho,self.R1,self.R2,self.tf,self.tlist)

        self.drho_dt = np.zeros([self.hdim,self.hdim,self.tsteps],dtype=np.complex128)
        self.drho_dt = tevol.drho_dt

        """
        Measure
        """   

        self.Mvec = np.zeros([3,self.tsteps],dtype=np.complex128)

        measuring = measure.measure(self.drho_dt,self.S_operator,self.tlist)

        self.Mvec = measuring.Mvec

        """
        Save data
        """   

        self.save_data()

        return


    def init_s_H(self):
     
        sH = hamiltonian.hamiltonian(self.B, self.S, self.dim, self.g_tensors, self.beta, self.J_tensors)
        self.S_operator = np.stack((sH.Sx,sH.Sy,sH.Sz),axis=-1)

        return sH.Hs

    

    def init_sp_coupling(self):

        """
        Compute the spin-phonon coupling energy.

        Parameters:
   
        Returns:    
        """
        R_vectors = np.random.rand(self.N_cells, 3)*10  # Random cell positions
        L1_vectors = np.random.rand(self.N_atoms, 3)*5  # Random displacement vectors
        L2_vectors = np.random.rand(self.N_atoms, 3)*5  # Random displacement vectors
        q_vector = np.random.rand(self.Nq,3)  # Mode indices (alpha, q)
        disp1 = np.linspace(-0.0025, 0.0025, 11)

        disp2 = np.array(list(itertools.product(disp1, disp1)))

        N_disp1 = len(disp1)
        N_disp2 = len(disp2)


        J_xi = np.zeros([self.N_atoms,3,N_disp1,self.Ns,self.Ns, 3, 3], dtype=np.complex128)
       
        J_xij = np.zeros([self.N_atoms,self.N_atoms,3,3,N_disp2,self.Ns,self.Ns, 3, 3], dtype=np.complex128)
       
        for n in range(self.N_atoms):
            for j in range(3):
                for i in range(N_disp1):
                    for k in range(self.Ns):
                        for l in range(self.Ns):
                            J_xi[n,j,i,k,l] = np.random.rand(3, 3)*20*4E-4

        for n in range(self.N_atoms):
            for m in range(self.N_atoms):
                for j in range(3):
                    for h in range(3):
                        for i in range(N_disp2):
                            for k in range(self.Ns):
                                for l in range(self.Ns):
                                    J_xij[n,m,j,h,i,k,l] = np.random.rand(3, 3)*20*4E-4

        V_q = coupling.coupling(self.dim, J_xi, J_xij, self.S, q_vector, self.omega_q, self.masses, R_vectors, L1_vectors,L2_vectors, disp1,disp2)

        self.V_alpha_q = V_q.V_alpha_q
        self.V_alpha_beta = np.random.rand(1)*(V_q.V_alpha_q) #randomize second derivative similar size to first derivative
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
            output.create_dataset('Mvec',data=self.Mvec)

        print("Data has been saved to data.h5")

        return

spin_phonon()

 

        
