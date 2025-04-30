import numpy as np
import hamiltonian
import coupling
import math_func
import redfield
import measure
import RK
import read_files
import h5py as h5
from constants import Bohrmagneton, k_B
import itertools

""""
Run file

Set input parameters

"""
class spin_phonon:
    def __init__(self, B,S, Delta_alpha_q,T,tf,dt, init_type='boltzmann'):
        
        """
        Spin Hamiltonian inputs

        """
        #Inputs

        self.B = B  # Magnetic field vector
        self.B = self.B * Bohrmagneton  # Convert to cm-1
        self.S = S  # Spin quantum number (spin-1/2)
        self.T = T  # Temperature in Kelvin
        self.init_type = init_type
        self.m = np.arange(-self.S, self.S+1, 1)
        self.Ns = int(2*self.S + 1)  # Number of spins
        self.hdim = self.Ns 
            
        self.q_vector, self.omega_q, self.L_vectors = read_files.read_phonons()
        self.R_vectors,self.masses,self.reciprocal_vectors = read_files.read_atoms()

          # Convert to eV

        self.q_vector = self.q_vector @ self.reciprocal_vectors # Convert q_vector to units of A^-1

        #Inputs
        
        self.N_atoms = len(self.masses)  # Number of atoms
        self.Nomega =  len(self.q_vector)  # Number of phonon modes
        self.Nq = self.q_vector.shape[0]  # Number of q points
        
        self.g_tensors,self.D_tensors = read_files.read_orca()

        #Outputs
        self.Hs = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.eigenvalues = np.zeros(self.hdim, dtype=np.complex128)
        self.eigenvectors = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.S_operator = np.zeros((self.Ns, self.Ns,3),dtype=np.complex128)


        self.Hs = self.init_s_H() #Zero displacement
        self.eigenvalues, self.eigenvectors = math_func.diagonalize(self.Hs)

        print("Eigenvalues of the spin Hamiltonian")
        print(self.eigenvalues)

        #Outputs
        self.V_alpha = np.zeros([self.Nq, self.Nomega,self.hdim, self.hdim],dtype=np.complex128)
    #    self.V_alpha_beta = np.zeros([self.hdim, self.hdim,self.Nq ,self.Nq],dtype=np.complex128)
    
    #    self.V_alpha_beta = np.tile(self.V_alpha[:, :, :, np.newaxis], (1, 1, 1, self.Nq))
        self.init_sp_coupling()

        

        """
        Green's function inputs
        """        
        #Inputs

        self.Delta_alpha_q = Delta_alpha_q  # Broadening parameter



        """
        Redfield equation
        """      
        self.R =np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
        self.R1 =np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
    #    self.R2 =np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
        self.R4 =np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)


        self.R1 = redfield.R1_tensor(self.V_alpha,self.eigenvalues,self.omega_q,self.Delta_alpha_q,self.T)
    #    self.R2 = redfield.R2_tensor(self.V_alpha,self.G_2ph,self.eigenvalues,self.omega_q,self.n_alpha_q,self.Delta_alpha_q)
        #self.R4 = redfield.R4_tensor(self.V_alpha,self.omega_q,self.eigenvalues,self.eigenvectors,self.n_alpha_q,self.Delta_alpha_q)

        self.R = self.R1


        self.R_mat = self.R.reshape((self.hdim**2, self.hdim**2))
        eigenvalues = np.linalg.eigvals(self.R_mat)
        print("Eigenvalues of the Redfield matrix")
        print(eigenvalues)
        

        #print("Eigenvectors of the Redfield matrix")
        #print(self.R_eigenvectors)


        """
        Initialize spin density
        """
        self.init_occ = np.zeros(self.hdim, dtype=np.complex128)
        self.rho0 = np.zeros([self.hdim**2], dtype=np.complex128)

        self.rho0 = self.init_rho()

       
        """
        Perform time-evolution
        """   

        #Inputs

        self.tf = tf  #Total time
        self.dt = dt  #Time step
        self.tlist = np.linspace(0, self.tf, int(self.tf / self.dt))
        self.tsteps = len(self.tlist)

        #Output

        self.drho_dt = np.zeros([self.tsteps,self.hdim**2],dtype=np.complex128)
        self.drho_dt = RK.RK(self.rho0,self.R_mat,self.dt,self.tlist)

        self.rho_t = np.zeros([self.hdim,self.hdim,self.tsteps],dtype=np.complex128)

        for t in range(self.tsteps):
            self.rho_t[:,:,t] = self.drho_dt[t].reshape(self.hdim,self.hdim)
        self.rho_t = 0.5*(self.rho_t + self.rho_t.conj().transpose(1,0,2)) # Ensure hermiticity
        self.rho_t = np.real(self.rho_t)  # Ensure hermiticity

        """
        Measure
        """   

        self.Mvec = np.zeros([3,self.tsteps],dtype=np.complex128)

        measuring = measure.measure(self.rho_t,self.S_operator,self.tlist)

        self.Mvec = measuring.Mvec

        self.T1 = measuring.T1
        self.T1_err = measuring.T1_err

        print("T1 = ",self.T1)
        print("T1_err = ",self.T1_err)

        """
        Save data
        """   

        self.save_data()

        return


    def init_s_H(self):
     
        sH = hamiltonian.hamiltonian(self.B, self.S, self.g_tensors, self.D_tensors)
        self.S_operator = np.stack((sH.Sx,sH.Sy,sH.Sz),axis=-1)

        return sH.Hs

    

    def init_sp_coupling(self):

        """
        Compute the spin-phonon coupling energy.

        Parameters:
   
        Returns:    
        """

        V_q = coupling.coupling(self.B, self.S, self.T, self.eigenvectors,self.q_vector, self.omega_q, self.masses, self.R_vectors, self.L_vectors)

        self.V_alpha = V_q.V_alpha
  
        return

    def init_rho(self):

        if self.init_type == 'polarized':
            state = np.zeros((self.hdim, 1), dtype=np.complex128)
            state[-1] = 1.0  # Last basis state corresponds to all spins down
            
            # Density matrix is outer product of the state
            rho0 = state @ state.conj().T

        elif self.init_type == 'boltzmann':
            if self.T == 0:
                # Zero temperature: all population in ground state
                self.init_occ[np.argmin(self.eigenvalues)] = 1.0
            else:
                beta = 1 / (k_B * self.T)
                self.init_occ = np.exp(-beta * self.eigenvalues)
                self.init_occ /= np.sum(self.init_occ)  # Normalize

            rho_diag = np.diag(self.init_occ.astype(np.complex128))
            rho0 = self.eigenvectors @ rho_diag @ self.eigenvectors.conj().T
        
        print(rho0)

        return rho0.flatten()
    

  
    def save_data(self):

        """
        Save data
        """

        with h5.File('data.h5', 'w') as f:
            input = f.create_group('input')
            input.create_dataset('tlist', data=self.tlist)

            output = f.create_group('output')
            output.create_dataset('rho_t', data=self.rho_t)
            output.create_dataset('Mvec',data=self.Mvec)


        print("Data has been saved to data.h5")

        return


 

        
