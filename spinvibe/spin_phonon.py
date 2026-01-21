import numpy as np
from . import hamiltonian, coupling, math_func, redfield, measure
import h5py as h5
from .constants import Bohrmagneton, k_B
import time
import scipy.linalg
from datetime import datetime

# Try to import MPI, fall back to serial mode if not available
try:
    from mpi4py import MPI
    mpi_on = True
except ImportError:
    mpi_on = False
    MPI = None

class spin_phonon:
    def __init__(self, B, S, Delta_alpha_q, rot_mat, pol, T, tf, dt, file_reader,save_file,init_type='polarized'):
        
        # Get MPI info if available, otherwise use serial mode
        if mpi_on:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            comm = None
            rank = 0
            size = 1

        init_time = time.perf_counter()

        if rank == 0:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("---------------------------------------")
            print("|Start Spin-phonon coupling simulation|")
            print("---------------------------------------")
            print("\n")

        timer_input = time.perf_counter()
        
        # Parameter initialization
        self.B_T = B  # Magnetic field vector
        self.Delta_alpha_q = Delta_alpha_q  # Broadening parameter
        self.S = S  # Spin quantum number 
        self.rot_mat = rot_mat
        self.pol = pol  # Polarization vector
        self.T = T  # Temperature in Kelvin
        self.init_type = init_type
        self.save_file = save_file
        if rank == 0:
            print("Input Parameters")
            print("================")
            print("Magnetic field:", self.B_T)
            print("S:", self.S)
            print("T:", self.T)  
            print("Broadening:", self.Delta_alpha_q)
            print("Population type:", self.init_type)
            print("Polarization:", self.pol)
            print("Rotational matrix:")
            print(self.rot_mat)
            print("\n")

            
        if rank == 0:
            self.open_output()

        self.B = self.B_T * Bohrmagneton  # Convert from T to cm-1
        self.m = np.arange(-self.S, self.S+1, 1)
        self.hdim = int(2*self.S + 1) 

        self.file_reader = file_reader
        self.q_vector, self.omega_q, self.L_vectors = self.file_reader.read_phonons()
        self.reciprocal_vectors = self.file_reader.read_atoms() 
        self.R_vectors = np.zeros(3)
        self.q_vector = self.q_vector @ self.reciprocal_vectors # Convert q vectors to A^-1

        self.N_atoms = len(self.q_vector)/3  # Number of atoms
        self.Nomega = len(self.q_vector)  # Number of phonon modes
        self.Nq = self.q_vector.shape[0]  # Number of q points
        
        g_tensor, d_tensor = self.file_reader.read_spin() #Read g-matrix and zero field splitting D-tensor

        #Rotate cartesian tensors to crystal coordinates
        self.g_tensor = self.rot_mat @ g_tensor @ self.rot_mat.T 
        self.d_tensor = self.rot_mat @ d_tensor @ self.rot_mat.T

        hours_input, minutes_input, seconds_input = self.timer(timer_input)

                        # Matrix initialization
        self.Hs = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.eigenvalues = np.zeros(self.hdim, dtype=np.complex128)
        self.eigenvectors = np.zeros([self.hdim, self.hdim], dtype=np.complex128)
        self.S_operator = np.zeros((self.hdim, self.hdim, 3), dtype=np.complex128)

        # Spin Hamiltonian setup
        self.Hs = self.init_s_H()  # Zero displacement spin Hamiltonian

        self.eigenvalues, self.eigenvectors = math_func.diagonalize(self.Hs)

        if rank == 0:
            print("Spin Hamiltonian")
            print(self.Hs)
            print("\n")
            print("Eigenvalues of the spin Hamiltonian")
            print(self.eigenvalues)
            print("\n")


        # Coupling matrices

        if rank == 0:
            print("Initialize simulation")
            print("=====================")

            if size > 1:
                print("Parallelizing computation across processes...")
                print(f"Number of processes: {size}")
                print("\n")
            else:
                print("Running in serial mode.")
                print("\n")

        # Initialize spin density

        self.init_occ = np.zeros(self.hdim, dtype=np.complex128)
        self.rho0 = np.zeros([self.hdim**2], dtype=np.complex128)
        self.rho0 = self.init_rho()

        init_Vq = coupling.coupling(self.B, self.S, self.T, self.eigenvectors,self.q_vector, self.omega_q, self.R_vectors, self.L_vectors,self.rot_mat,self.file_reader,self.save_file)

        #Initialize Redfield superoperator

        self.init_R = redfield.Redfield(self.S, self.T, self.eigenvectors,self.eigenvalues,self.q_vector, self.omega_q,self.Delta_alpha_q, self.L_vectors)

        hours_input, minutes_input, seconds_input = self.timer(timer_input)

        # Redfield tensors
        self.R = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
        self.R1 = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
        self.R2 = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)

        if rank == 0:
            print("Initializing R1 tensor (Linear coupling)")

        timer_R1 = time.perf_counter()
        self.R1 = self.init_R.R1_tensor(init_Vq)
        hours_R1, minutes_R1, seconds_R1 = self.timer(timer_R1)

        if rank == 0:
            R1_mat = self.R1.reshape((self.hdim**2, self.hdim**2)) #Transform into matrix form

            eigenvalues, eigenvectors = np.linalg.eig(R1_mat)
            print("Eigenvalues of the R1 matrix")
            print(eigenvalues)
            print("\n")
            
            self.tf = tf  # Total time
            self.dt = dt  # Time step
            self.tlist = np.linspace(0, self.tf, int(self.tf / self.dt))
            self.tsteps = len(self.tlist)

            self.drho_dt = self.time_evolution(eigenvalues, eigenvectors)

            self.rho_t = np.zeros([self.hdim, self.hdim, self.tsteps], dtype=np.complex128)

            self.rho_t = self.drho_dt.reshape(self.tsteps, self.hdim, self.hdim).transpose(2, 1, 0)


            self.Mz = np.zeros([self.tsteps], dtype=np.complex128)

            measuring = measure.measure(self.rho_t, self.S_operator,self.tlist, self.pol,self.init_type)

            self.Mz = measuring.Mz
            self.T1 = measuring.T1
            self.T1_err = measuring.T1_err
        
            print("Linear coupling T1")
            print("T1 = ", self.T1,"s")
            print("T1_err = ", self.T1_err,"s")
            if self.T1 == 1 or self.T1_err == 0:
                print("Warning: T1 likely fitting failed!!! Please check M(t) data")
            if self.T1 < self.T1_err:
                print("Warning: T1_err is larger than T1!!! Fitting likely failed!!!")
            print("\n")



        if rank == 0:
            print("Initializing R2 tensor (Quadratic coupling)")

        timer_R2 = time.perf_counter()
        self.R2 = self.init_R.R2_tensor(init_Vq)
        hours_R2, minutes_R2, seconds_R2 = self.timer(timer_R2)        

        if rank == 0:
            R2_mat = self.R2.reshape((self.hdim**2, self.hdim**2)) #Transform into matrix form

            eigenvalues, eigenvectors = np.linalg.eig(R2_mat)
            print("Eigenvalues of the R2 matrix")
            print(eigenvalues)
            print("\n")
    
            
            self.tf = tf  # Total time
            self.dt = dt  # Time step
            self.tlist = np.linspace(0, self.tf, int(self.tf / self.dt))
            self.tsteps = len(self.tlist)

            self.drho_dt = self.time_evolution(eigenvalues, eigenvectors)
            self.rho_t = np.zeros([self.hdim, self.hdim, self.tsteps], dtype=np.complex128)
            self.rho_t = self.drho_dt.reshape(self.tsteps, self.hdim, self.hdim).transpose(2, 1, 0)

            self.Mz = np.zeros([self.tsteps], dtype=np.complex128)

            measuring = measure.measure(self.rho_t, self.S_operator,self.tlist, self.pol,self.init_type)

            self.Mz = measuring.Mz
            self.T1 = measuring.T1
            self.T1_err = measuring.T1_err
        
            print("Quadratic coupling T1")
            print("T1 = ", self.T1,"s")
            print("T1_err = ", self.T1_err,"s")
            if self.T1 == 1 or self.T1_err == 0:
                print("Warning: T1 likely fitting failed!!! Please check M(t) data")
            if self.T1 < self.T1_err:
                print("Warning: T1_err is larger than T1!!! Fitting likely failed!!!")
            print("\n")


        self.R = self.R1 + self.R2
            
        self.R_mat = np.zeros((self.hdim**2, self.hdim**2), dtype=np.complex128)
        self.R_mat = self.R.reshape((self.hdim**2, self.hdim**2))

        eigenvalues, eigenvectors = np.linalg.eig(self.R_mat)
    

        if rank == 0:
            print("Eigenvalues of the Redfield matrix")
            print(eigenvalues)
            print("\n")

        # Time evolution and measurement
        if rank == 0:
        
            timer_evol = time.perf_counter()
            
            self.tf = tf  # Total time
            self.dt = dt  # Time step
            self.tlist = np.linspace(0, self.tf, int(self.tf / self.dt))
            self.tsteps = len(self.tlist)

            self.drho_dt = self.time_evolution(eigenvalues, eigenvectors)

            self.rho_t = np.zeros([self.hdim, self.hdim, self.tsteps], dtype=np.complex128)

            self.rho_t = self.drho_dt.reshape(self.tsteps, self.hdim, self.hdim).transpose(2, 1, 0)

            hours_evol, minutes_evol, seconds_evol = self.timer(timer_evol)

            #Compute magnetization evolution
            timer_measure = time.perf_counter()

            self.Mz = np.zeros([self.tsteps], dtype=np.complex128)

            measuring = measure.measure(self.rho_t, self.S_operator,self.tlist, self.pol,self.init_type)

            self.Mz = measuring.Mz
            self.T1 = measuring.T1
            self.T1_err = measuring.T1_err

            hours_measure, minutes_measure, seconds_measure = self.timer(timer_measure)
        
            print("Total T1 from magnetization decay")
            print("T1 = ", self.T1,"s")
            print("T1_err = ", self.T1_err,"s")
            if self.T1 == 1 or self.T1_err == 0:
                print("Warning: T1 likely fitting failed!!! Please check M(t) data")
            if self.T1 < self.T1_err:
                print("Warning: T1_err is larger than T1!!! Fitting likely failed!!!")
            print("\n")

            print("Saving data")
            self.save_data()
            print("\n")

        
        if rank == 0:
            hours, minutes, seconds = self.timer(init_time)
            print(f"Initiate simulation: {hours_input}h {minutes_input}m {seconds_input:.2f}s")
            print(f"Build R1: {hours_R1}h {minutes_R1}m {seconds_R1:.2f}s")
            print(f"Build R2: {hours_R2}h {minutes_R2}m {seconds_R2:.2f}s")
            print(f"Time evolution: {hours_evol}h {minutes_evol}m {seconds_evol:.2f}s")
            print(f"Measuring Time: {hours_measure}h {minutes_measure}m {seconds_measure:.2f}s")
            print(f"Total Run Time: {hours}h {minutes}m {seconds:.2f}s")

        return

    def init_s_H(self):
     
        sH = hamiltonian.hamiltonian(self.B, self.S, self.g_tensor, self.d_tensor)
        self.S_operator = np.stack((sH.Sx,sH.Sy,sH.Sz),axis=-1)

        return sH.Hs



    def init_rho(self):
        
        # Get MPI info if available, otherwise use serial mode
        if mpi_on:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            rank = 0
            size = 1
    
        if self.init_type == 'polarized':
            # polarization direction (default z-axis)
            n = np.array(self.pol if self.pol is not None else [0,0,1], dtype=float)
            n /= np.linalg.norm(n)

            # extract spin operators
            Sx = self.S_operator[:,:,0]
            Sy = self.S_operator[:,:,1]  
            Sz = self.S_operator[:,:,2]

            # Identity matrix
            I = np.eye(self.hdim, dtype=complex)
            
            if self.S == 0:
                rho0 = I
            else:
                # General spin case
                rho0 = (1/self.hdim) * (I + (n[0] * Sx + n[1] * Sy + n[2] * Sz))
            
            # Ensure hermiticity and proper normalization
            rho0 = 0.5 * (rho0 + rho0.conj().T)
            rho0 = rho0 / np.trace(rho0)

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

        elif self.init_type == 'pure':
                # Find degenerate highest energy states
            max_energy = np.max(self.eigenvalues)
            tolerance = 1e-3
            degenerate_indices = np.where(np.abs(self.eigenvalues - max_energy) < tolerance)[0]
        
            
            # Initialize density matrix directly in the computational basis
            rho0 = np.zeros((self.hdim, self.hdim), dtype=complex)
            
            # Split population equally among degenerate BASIS states
            weight = 1.0 / len(degenerate_indices)
            for idx in degenerate_indices:
                rho0[idx, idx] = weight 
        
        if rank == 0:   
            print("Initial spin population:")
            print(rho0)
            print("\n")

        return rho0.flatten()

    def time_evolution(self,eigenvalues, eigenvectors):
        """
        Time evolution using propagator method
        """
        
        nsteps = len(self.tlist)
        hdim = len(self.rho0)
        rho = np.zeros((nsteps, hdim), dtype=np.complex128)
        rho[0] = self.rho0.copy()  # Initial state
        
        # Compute inverse of eigenvector matrix (V^{-1} in the equation)
        V_inv = np.linalg.inv(eigenvectors)
        
        # Time evolution loop
        for i in range(1, nsteps):
            # Current time
            t = self.tlist[i]
            
            # Construct propagator L(t) according to equation (188)
            # L_ij(t) = sum_k V_ik * exp(lambda_k * t) * V_kj^{-1}
            exp_diag = np.exp(eigenvalues * t)  # Exponential of eigenvalues
            L_t = eigenvectors @ np.diag(exp_diag) @ V_inv
            
            # Apply propagator to initial density matrix (equation 189)
            # rho_i(t) = sum_j L_ij(t) * rho_j(t=0)
            rho[i] = L_t @ rho[0]
        
        return rho
    
    def open_output(self):

        with h5.File(self.save_file, 'w') as f:
            input = f.create_group('input')
            output = f.create_group('output')

        return

    
    def save_data(self):

        """
        Save data

            - tlist: array of time points
            - redfield_matrix: Redfield matrix
            - rhot_t: Time evolution of the spin density
            - Mvec: Time evolution of the magnetization
        """

        with h5.File(self.save_file, 'a') as f:
            input = f['input']
            output = f['output']

            input.create_dataset('tlist', data=self.tlist)

            output.create_dataset('redfield_matrix', data=self.R_mat)
            output.create_dataset('rho_t', data=self.rho_t)
            output.create_dataset('M',data=self.Mz)

        self.init_R.save_data(self.save_file)

        print(f"Data has been saved to {self.save_file}")

        return

    def timer(self,start_time):
        total_time = time.perf_counter() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60

        return hours,minutes,seconds