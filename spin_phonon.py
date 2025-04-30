import numpy as np
import hamiltonian
import coupling
import math_func
import redfield
import measure
import read_files
import h5py as h5
from constants import Bohrmagneton, k_B
import time

""""
Run file

Set input parameters

"""
class spin_phonon:
    def __init__(self, B,S, Delta_alpha_q,T,tf,dt, init_type='boltzmann'):
        
        init_time = time.perf_counter()

        print("Start Spin-phonon coupling simulation")

        print("\n")

        timer_input = time.perf_counter()
        """
        Spin Hamiltonian inputs

        """
        #Inputs

        self.B = B  # Magnetic field vector
        self.Delta_alpha_q = Delta_alpha_q  # Broadening parameter
        self.S = S  # Spin quantum number 
        self.T = T  # Temperature in Kelvin
        self.init_type = init_type

        
        print("Input Parameters:")
 
        print("Magnetic field:", self.B)
        print("S:",self.S)
        print("T:",self.T)
        print("Broadening:",self.Delta_alpha_q)
        print("Population type:",self.init_type)
        

        self.B = self.B * Bohrmagneton  # Convert to cm-1


        self.m = np.arange(-self.S, self.S+1, 1)
        self.Ns = int(2*self.S + 1)  # Number of spins
        self.hdim = self.Ns 

        
        
        self.q_vector, self.omega_q, self.L_vectors = read_files.read_phonons()
        self.R_vectors,self.reciprocal_vectors = read_files.read_atoms()

          # Convert to eV

        self.q_vector = self.q_vector @ self.reciprocal_vectors # Convert q_vector to units of A^-1

        #Inputs
        
        self.N_atoms = self.R_vectors.shape[0]  # Number of atoms
        self.Nomega =  len(self.q_vector)  # Number of phonon modes
        self.Nq = self.q_vector.shape[0]  # Number of q points
        
        self.g_tensors,self.D_tensors = read_files.read_orca()

        
        hours_input,minutes_input,seconds_input = self.timer(timer_input)

        
        print("\n")
        print("Output:")


        timer_redfield = time.perf_counter()

        #Outputs
        self.Hs = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.eigenvalues = np.zeros(self.hdim, dtype=np.complex128)
        self.eigenvectors = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.S_operator = np.zeros((self.Ns, self.Ns,3),dtype=np.complex128)


        self.Hs = self.init_s_H() #Zero displacement
        self.eigenvalues, self.eigenvectors = math_func.diagonalize(self.Hs)

        print("Eigenvalues of the spin Hamiltonian")
        print(self.eigenvalues)
        print("\n")

        

        #Outputs
        self.V_alpha = np.zeros([self.Nq, self.Nomega,self.hdim, self.hdim],dtype=np.complex128)
    #    self.V_alpha_beta = np.zeros([self.hdim, self.hdim,self.Nq ,self.Nq],dtype=np.complex128)
    
    #    self.V_alpha_beta = np.tile(self.V_alpha[:, :, :, np.newaxis], (1, 1, 1, self.Nq))
        self.init_sp_coupling()

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

        
        hours_redfield,minutes_redfield,seconds_redfield = self.timer(timer_redfield)

        

        
        """
        Initialize spin density
        """
        print("\n")

        self.init_occ = np.zeros(self.hdim, dtype=np.complex128)
        self.rho0 = np.zeros([self.hdim**2], dtype=np.complex128)

        self.rho0 = self.init_rho()

        
        """
        Perform time-evolution
        """   
        print("\n")
        
        timer_evol = time.perf_counter()
        #Inputs

        self.tf = tf  #Total time
        self.dt = dt  #Time step
        self.tlist = np.linspace(0, self.tf, int(self.tf / self.dt))
        self.tsteps = len(self.tlist)

        #Output

        self.drho_dt = np.zeros([self.tsteps,self.hdim**2],dtype=np.complex128)
        self.drho_dt = self.RK(self.rho0,self.R_mat,self.dt,self.tlist)

        self.rho_t = np.zeros([self.hdim,self.hdim,self.tsteps],dtype=np.complex128)

        for t in range(self.tsteps):
            self.rho_t[:,:,t] = self.drho_dt[t].reshape(self.hdim,self.hdim)
        self.rho_t = 0.5*(self.rho_t + self.rho_t.conj().transpose(1,0,2)) # Ensure hermiticity
        self.rho_t = np.real(self.rho_t)  # Ensure hermiticity

        hours_evol,minutes_evol,seconds_evol = self.timer(timer_evol)


        """
        Measure
        """   
        timer_measure = time.perf_counter()

        self.Mvec = np.zeros([3,self.tsteps],dtype=np.complex128)

        measuring = measure.measure(self.rho_t,self.S_operator,self.tlist)

        self.Mvec = measuring.Mvec

        self.T1 = measuring.T1
        self.T1_err = measuring.T1_err

        hours_measure,minutes_measure,seconds_measure = self.timer(timer_measure)
        
        print("T1 = ",self.T1)
        print("T1_err = ",self.T1_err)

        """
        Save data
        """   
        print("\n")

        self.save_data()

        print("\n")

        hours,minutes,seconds = self.timer(init_time)
        print(f"Read input Time: {hours_input}h {minutes_input}m {seconds_input:.2f}s")
        print(f"Build Redfield: {hours_redfield}h {minutes_redfield}m {seconds_redfield:.2f}s")
        print(f"Time evolution: {hours_evol}h {minutes_evol}m {seconds_evol:.2f}s")
        print(f"Measuring Time: {hours_measure}h {minutes_measure}m {seconds_measure:.2f}s")
        print(f"Total Run Time: {hours}h {minutes}m {seconds:.2f}s")

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

        V_q = coupling.coupling(self.B, self.S, self.T, self.eigenvectors,self.q_vector, self.omega_q, self.R_vectors, self.L_vectors)

        self.V_alpha = V_q.V_alpha
  
        return

    def init_rho(self):

        if self.init_type == 'polarized':
            state = np.zeros((self.hdim, 1), dtype=np.complex128)
            state[-1] = 1.0  # Along z direction
            
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
        
        print("Initial spin population:")
        print(rho0)

        return rho0.flatten()


        

    def RK(self):

        from scipy.integrate import solve_ivp

        nsteps = len(self.tlist)
        hdim = len(self.rho0)
        rho = np.zeros((nsteps,hdim), dtype=np.complex128)
        rho[0] = self.rho0.copy()  # Force rho0 to be 1D
        
        for i in range(nsteps - 1):
            # Ensure R is 2D and rho[i] is 1D
            k1 = self.R_mat @ rho[i]
            k2 = self.R_mat @ (rho[i] + 0.5 * self.dt * k1)
            k3 = self.R_mat @ (rho[i] + 0.5 * self.dt * k2)
            k4 = self.R_mat @ (rho[i] + self.dt * k3)
            
            rho[i+1] = rho[i] + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return rho
    
    def save_data(self):

        """
        Save data
        """

        with h5.File('Redfield_SPh_coupling.h5', 'w') as f:
            input = f.create_group('input')
            input.create_dataset('tlist', data=self.tlist)

            output = f.create_group('output')
            output.create_dataset('rho_t', data=self.rho_t)
            output.create_dataset('Mvec',data=self.Mvec)


        print("Data has been saved to Redfield_SPh_coupling.h5")

        return

    def timer(self,start_time):
        total_time = time.perf_counter() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60

        return hours,minutes,seconds

 

        
