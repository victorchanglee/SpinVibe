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
from mpi4py import MPI


class spin_phonon:
    def __init__(self, B, S, Delta_alpha_q, T, tf, dt, init_type='boltzmann'):
        
        # MPI setup
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        init_time = time.perf_counter()

        if rank == 0:
            print("Start Spin-phonon coupling simulation")
            print("\n")

        timer_input = time.perf_counter()
        
        # Parameter initialization (ALL ranks)
        self.B0 = B
        self.B = B  # Magnetic field vector
        self.Delta_alpha_q = Delta_alpha_q  # Broadening parameter
        self.S = S  # Spin quantum number 
        self.T = T  # Temperature in Kelvin
        self.init_type = init_type

        if rank == 0:
            print("Input Parameters:")
            print("Magnetic field:", self.B)
            print("S:", self.S)
            print("T:", self.T)
            print("Broadening:", self.Delta_alpha_q)
            print("Population type:", self.init_type)

        # Unit conversions and array setup (ALL ranks)
        self.B = self.B * Bohrmagneton  # Convert to cm-1
        self.m = np.arange(-self.S, self.S+1, 1)
        self.Ns = int(2*self.S + 1)  # Number of spins
        self.hdim = self.Ns 

        # File reading (ALL ranks need this data)
        self.q_vector, self.omega_q, self.L_vectors = read_files.read_phonons()
        self.R_vectors, self.reciprocal_vectors = read_files.read_atoms()
        self.q_vector = self.q_vector @ self.reciprocal_vectors # Convert q_vector to units of A^-1

        # More parameter setup (ALL ranks)
        self.N_atoms = self.R_vectors.shape[0]  # Number of atoms
        self.Nomega = len(self.q_vector)  # Number of phonon modes
        self.Nq = self.q_vector.shape[0]  # Number of q points
        
        self.g_tensors, self.D_tensors = read_files.read_orca()

        hours_input, minutes_input, seconds_input = self.timer(timer_input)

        if rank == 0:
            print("\n")
            print("Output:")

        # Matrix initialization (ALL ranks)
        self.Hs = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.eigenvalues = np.zeros(self.hdim, dtype=np.complex128)
        self.eigenvectors = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.S_operator = np.zeros((self.Ns, self.Ns, 3), dtype=np.complex128)

        # Spin Hamiltonian setup (ALL ranks)
        self.Hs = self.init_s_H()  # Zero displacement
        self.eigenvalues, self.eigenvectors = math_func.diagonalize(self.Hs)

        if rank == 0:
            print("Eigenvalues of the spin Hamiltonian")
            print(self.eigenvalues)
            print("\n")

        # Coupling matrices (ALL ranks)

        if rank == 0:
            print("Initialize simulation")

        self.V_alpha = np.zeros([self.Nq, self.Nomega, self.hdim, self.hdim], dtype=np.complex128)
    
        init_Vq = coupling.coupling(self.B, self.S, self.T, self.eigenvectors,self.q_vector, self.omega_q, self.R_vectors, self.L_vectors)

        self.V_alpha = init_Vq.V_alpha  # Coupling matrix for phonon modes

        init_R = redfield.Redfield(self.S, self.T, self.eigenvectors,self.eigenvalues,self.q_vector, self.omega_q,self.Delta_alpha_q, self.L_vectors)

        hours_input, minutes_input, seconds_input = self.timer(timer_input)

        # Redfield tensors (ALL ranks)
        self.R = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
        self.R1 = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
        self.R2 = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
        #self.R4 = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)

        if rank == 0:
            print("Initializing R1 tensor")

        timer_R1 = time.perf_counter()
        self.R1 = init_R.R1_tensor(self.V_alpha)
        hours_R1, minutes_R1, seconds_R1 = self.timer(timer_R1)
        R1_mat = self.R1.reshape((self.hdim**2, self.hdim**2))

    
        eigenvalues = np.linalg.eigvals(R1_mat)

        if rank == 0:
            print("Eigenvalues of the R1 tensor")
            print(eigenvalues)
            print("\n")
            print("Initializing R2 tensor")

        timer_R2 = time.perf_counter()
        self.R2 = init_R.R2_tensor(init_Vq)
        eigenvalues = np.linalg.eigvals(self.R2.reshape((self.hdim**2, self.hdim**2)))
        hours_R2, minutes_R2, seconds_R2 = self.timer(timer_R2)

        

        if rank == 0:
            print("Eigenvalues of the R2 tensor")
            print(eigenvalues)
            print("\n")


        #timer_R4 = time.perf_counter()
        #self.R4 = init_R.R4_tensor(self.V_alpha)
        #hours_R4, minutes_R4, seconds_R4 = self.timer(timer_R4)

        #eigenvalues = np.linalg.eigvals(self.R4.reshape((self.hdim**2, self.hdim**2)))

        #if rank == 0:
        #    print("Eigenvalues of the R4 tensor")
        #    print(eigenvalues)
        #    print(f"Build R4: {hours_R4}h {minutes_R4}m {seconds_R4:.2f}s")
        #    print("\n")
        
        self.R = self.R1 + self.R2 #+ self.R4

        self.R_mat = np.zeros((self.hdim**2, self.hdim**2), dtype=np.complex128)
        self.R_mat = self.R.reshape((self.hdim**2, self.hdim**2))

        eigenvalues = np.linalg.eigvals(self.R_mat)
        
        if rank == 0:
            print("Eigenvalues of the Redfield matrix")
            print(eigenvalues)

        # Initialize spin density (ALL ranks)
        if rank == 0:
            print("\n")

        self.init_occ = np.zeros(self.hdim, dtype=np.complex128)
        self.rho0 = np.zeros([self.hdim**2], dtype=np.complex128)
        self.rho0 = self.init_rho()

        # Time evolution and measurement (Only rank 0)
        if rank == 0:
            print("Start time evolution")
            print("\n")
        
            timer_evol = time.perf_counter()
            
            self.tf = tf  # Total time
            self.dt = dt  # Time step
            self.tlist = np.linspace(0, self.tf, int(self.tf / self.dt))
            self.tsteps = len(self.tlist)

            self.drho_dt = np.zeros([self.tsteps, self.hdim**2], dtype=np.complex128)
            self.drho_dt = self.RK()

            self.rho_t = np.zeros([self.hdim, self.hdim, self.tsteps], dtype=np.complex128)

            for t in range(self.tsteps):
                self.rho_t[:, :, t] = self.drho_dt[t].reshape(self.hdim, self.hdim)

            hours_evol, minutes_evol, seconds_evol = self.timer(timer_evol)

            timer_measure = time.perf_counter()

            self.Mvec = np.zeros([3, self.tsteps], dtype=np.complex128)

            measuring = measure.measure(self.rho_t, self.S_operator, self.tlist)

            self.Mvec = measuring.Mvec
            self.T1 = measuring.T1
            self.T1_err = measuring.T1_err

            hours_measure, minutes_measure, seconds_measure = self.timer(timer_measure)
        
       
            print("T1 = ", self.T1)
            print("T1_err = ", self.T1_err)

        
            print("\n")
            self.save_data()
            print("\n")

        
        if rank == 0:
            hours, minutes, seconds = self.timer(init_time)
            print(f"Initiate simulation: {hours_input}h {minutes_input}m {seconds_input:.2f}s")
            print(f"Build R1: {hours_R1}h {minutes_R1}m {seconds_R1:.2f}s")
            print(f"Build R2: {hours_R2}h {minutes_R2}m {seconds_R2:.2f}s")
            #print(f"Build R4: {hours_R4}h {minutes_R4}m {seconds_R4:.2f}s")
            print(f"Time evolution: {hours_evol}h {minutes_evol}m {seconds_evol:.2f}s")
            print(f"Measuring Time: {hours_measure}h {minutes_measure}m {seconds_measure:.2f}s")
            print(f"Total Run Time: {hours}h {minutes}m {seconds:.2f}s")

        return

    def init_s_H(self):
     
        sH = hamiltonian.hamiltonian(self.B, self.S, self.g_tensors, self.D_tensors)
        self.S_operator = np.stack((sH.Sx,sH.Sy,sH.Sz),axis=-1)

        return sH.Hs



    def init_rho(self):
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    
        if self.init_type == 'polarized':
            state = np.zeros((self.hdim, 1), dtype=np.complex128)
            state[-1] = 1.0  # Along z direction
            
            # Density matrix is outer product of the state
            rho0 = state @ state.conj().T

        elif self.init_type == 'inverted':
            # Put all population in highest energy state
            rho0 = np.zeros((self.hdim, self.hdim), dtype=np.complex128)
            rho0[0, 0] = 1.0  # All in |ms = -1⟩ (if this is highest energy)

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
        
        if rank == 0:   
            print("Initial spin population:")
            print(rho0)
            print("\n")

        return rho0.flatten()


        

    def RK(self):

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

        with h5.File(f"Spin_phonon_{self.B0[0]}.{self.B0[1]}.{self.B0[2]}T_{self.T}K_{self.Delta_alpha_q}.h5", 'w') as f:
            input = f.create_group('input')
            input.create_dataset('tlist', data=self.tlist)

            output = f.create_group('output')
            output.create_dataset('redfield_matrix', data=self.R_mat)
            output.create_dataset('rho_t', data=self.rho_t)
            output.create_dataset('Mvec',data=self.Mvec)


        print(f"Data has been saved to Spin_phonon_{self.B0[0]}.{self.B0[1]}.{self.B0[2]}T_{self.T}K_{self.Delta_alpha_q}.h5")

        return

    def timer(self,start_time):
        total_time = time.perf_counter() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60

        return hours,minutes,seconds

 

        
