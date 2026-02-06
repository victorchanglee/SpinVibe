import numpy as np
from .constants import hbar_SI, c_cms
from . import hamiltonian
from . import math_func
import h5py as h5

# Try to import MPI, fall back to serial mode if not available
try:
    from mpi4py import MPI
    mpi_on = True
except ImportError:
    mpi_on = False
    MPI = None

class coupling:
    def __init__(self, B, S, T, eigenvectors, q_vector, omega_q, R_vectors, L_vectors,masses, rot_mat, disp1, disp2, file_reader,save_file):

        # Get MPI info if available, otherwise use serial mode
        if mpi_on:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            comm = None
            rank = 0
            size = 1
    
        self.file_reader = file_reader
        self.B = B # Magnetic field in cm-1
        self.S = S # Spin
        self.T = T # Temperatur in K
        self.Ns = int(2*self.S + 1) # Number of spins
        self.hdim = self.Ns
        self.eigenvectors = eigenvectors # Eigenvectors of the spin Hamiltonian
        self.indices = self.file_reader.read_indices() # Mapping of the molecule indices in the crystal
        self.N = len(self.indices) # Number of atoms in the molecule
        self.masses = masses[self.indices] # Atomic mass of the molecule

        self.q_vector = q_vector # Phonon mode index and wave vector q 
        self.omega_q = omega_q # Phonon frequencies in cm-1
        self.Nq = L_vectors.shape[0] # Number of q points
        self.Nomega = L_vectors.shape[1] # Number of phonons at q 

        self.R_vectors_mol = np.zeros((self.N, 3), dtype=np.complex128)
        self.R_vectors_mol = R_vectors[self.indices,:] # Lattice vectors
        self.L_vectors_mol = np.zeros((self.Nq, self.Nomega, self.N, 3), dtype=np.complex128) # Eigenvectors of the molecule in the crystal
        self.L_vectors_mol = L_vectors[:,:,self.indices,:] 
        self.rot_mat = rot_mat # Rotational matrix for hte molecule to match the coordinates in the crystal
        self.disp1 = disp1
        self.disp2 = np.stack([disp2,disp2], axis=0) # Displacements for finite difference
        
        d_tensor, g_tensor = self.file_reader.read_spin() # g-matrix and ZFS D-tensor

        #Rotate the molecular cartesian matrices to match the crystal coordinates
        self.g_tensor = self.rot_mat @ g_tensor @ self.rot_mat.T
        self.d_tensor = self.rot_mat @ d_tensor @ self.rot_mat.T

        #Read linear displacement g-matrix and ZFS D-tensor
        D_d1, G_d1, D_d2, G_d2 = self.file_reader.read_derivatives()

        self.G_d1 = self.rot_mat @ G_d1 @ self.rot_mat
        self.D_d1 = self.rot_mat @ D_d1 @ self.rot_mat
        self.G_d2 = self.rot_mat @ G_d2 @ self.rot_mat
        self.D_d2 = self.rot_mat @ D_d2 @ self.rot_mat

        self.dH_dx = np.zeros((self.N,3,self.hdim,self.hdim), dtype=np.complex128)
        self.compute_dH_dx() #Compute dH/dx

        self.dH2_dxdx = np.zeros((self.N,self.N,3,3,self.hdim,self.hdim), dtype=np.complex128)
        self.compute_d2H_dxdx() #Compute d2H/dxdx'

        if rank == 0:
            self.save_file = save_file
            self.save_data()

        #Initizialize quadratic spin-phonon coupling
        self.pre_compute_V_alpha_beta_q()

        return 

    
    def compute_dH_dx(self):
        # Compute the first derivative of the Hamiltonian with respect to atomic displacements  
        # second index of G_d1

        dg = np.zeros((3, self.N, 3, 3), dtype=np.float64)
        dd = np.zeros((3,  self.N, 3, 3), dtype=np.float64)

        #Loop over each dimension except atomic displacement
        for i in range(3):
            for atom in range(self.N):
                for j in range(3):
                    for k in range(3):
                        g_x = self.G_d1[i,atom,:,j,k]
                        d_x = self.D_d1[i,atom,:,j,k]
                        
                        #Pass matrix entries as a function of the atomic displacement to compute the derivatives
                        dg[i,atom,j,k] = math_func.compute_derivative(self.disp1, g_x)
                        dd[i,atom,j,k] = math_func.compute_derivative(self.disp1, d_x)
                            
        #Compute the derivative of the Hamiltonian
        for n in range( self.N):
            for i in range(3):
                dg_dxi = dg[i,n, :, :]
                dD_dxi = dd[i,n, :, :]
                sH = hamiltonian.hamiltonian( self.B, self.S, dg_dxi, dD_dxi)

                self.dH_dx[n,i,:,:] = sH.Hs



    def compute_d2H_dxdx(self):
        # Compute the second derivative of the Hamiltonian with respect to atomic displacements  

        d2g = np.zeros((self.N, self.N,3,3, 3, 3), dtype=np.float64)
        d2d = np.zeros((self.N, self.N,3,3, 3, 3), dtype=np.float64)

        for atom1 in range(self.N):
            for atom2 in range(self.N):
                for i in range(3):
                    for j in range(3):
                        if atom1 == atom2:
                            continue
                        else:
                            for k in range(3):
                                for l in range(3):
                                    g_x = self.G_d2[atom1,atom2,i,j,:,:,k,l]
                                    d_x = self.D_d2[atom1,atom2,i,j,:,:,k,l]

                                    d2g[atom1,atom2,i,j,k,l] = math_func.compute_second_derivative(self.disp2, g_x)
                                    d2d[atom1,atom2,i,j,k,l] = math_func.compute_second_derivative(self.disp2, d_x)

        for atom1 in range( self.N):
            for atom2 in range(self.N):
                for i in range(3):
                    for j in range(3):
                        if atom1 == atom2:
                            continue
                        else:
                            d2g_dxij = d2g[atom1,atom2,i,j,:,:]
                            d2D_dxij = d2d[atom1,atom2,i,j,:,:]
                
                            sH = hamiltonian.hamiltonian( self.B, self.S, d2g_dxij, d2D_dxij)
                            self.dH2_dxdx[atom1,atom2,i,j,:,:] = sH.Hs

    def compute_V_alpha_q(self, q, omega):
        """
        Compute the interaction matrix elements V^{alpha q}_{aj}.
        """

        # Initialize local temporary array
        tmp = np.zeros((self.N,3), dtype=np.complex128)

         # Phase factor from the lattice vectors

        for atom in range(self.N):
            freq = 2 * np.pi * self.omega_q[q,omega] * c_cms # Convert to radian/s

            if freq <= 0:
                prefactor = 0.0
            else:
                prefactor = np.sqrt(hbar_SI / (self.Nq * freq * self.masses[atom])) 
                prefactor *= 1E10  # Convert to A units

            exp = np.exp(1j * np.dot(self.q_vector[q,:], self.R_vectors_mol[atom,:]))

            tmp[atom] = prefactor * exp * self.L_vectors_mol[q, omega, atom]

        coupling = np.einsum('il,ilab -> ab', tmp, self.dH_dx,optimize=True)

        H_herm = np.conj(coupling.T)
        V_alpha = 0.5 * (coupling + H_herm)

        return V_alpha

    def pre_compute_V_alpha_beta_q(self):
        
        self.w = 2 * np.pi * self.omega_q * c_cms

        base_factor = np.sqrt(hbar_SI / (self.Nq )) 
    
        self.prefactor = np.zeros((self.Nq, self.Nomega, self.N))
        
        for nq in range(self.Nq):
            for nomega in range(self.Nomega):
                w_val = self.w[nq, nomega]
                if w_val <= 0:
                    self.prefactor[nq, nomega, :] = 0.0
                else:
                    self.prefactor[nq, nomega, :] = base_factor / np.sqrt(w_val * self.masses)  
                    self.prefactor[nq, nomega, :] *= 1E10  # Convert to A units

        i_indices = np.arange(self.N)
        self.valid_i = np.repeat(i_indices, self.N-1)
        self.valid_j = np.concatenate([np.arange(i) for i in range(self.N)] + 
                                [np.arange(i+1, self.N) for i in range(self.N)])

        return

    def compute_V_alpha_beta_q(self, Nq1, Nq2, Nomega1, Nomega2):

        

        q1_dot_R1 = np.dot(self.R_vectors_mol, self.q_vector[Nq1,:])
        q2_dot_R2 = np.dot(self.R_vectors_mol, self.q_vector[Nq2,:])

        exp1 = np.exp(1j * q1_dot_R1)
        exp2 = np.exp(1j * q2_dot_R2)

        prefactor1 = self.prefactor[Nq1, Nomega1, :]  * exp1
        prefactor2 = self.prefactor[Nq2, Nomega2, :]  * exp2

        L1 = self.L_vectors_mol[Nq1, Nomega1, :, :] 
        L2 = self.L_vectors_mol[Nq2, Nomega2, :, :]  
    
        #combined_factors1 = np.einsum('n,i -> ni',prefactor1,exp1)
        #combined_factors2 = np.einsum('n,i -> ni',prefactor2,exp2)
        
        tmp1 = np.einsum('n,ni->ni', prefactor1, L1, optimize=True)
        tmp2 = np.einsum('n,ni->ni', prefactor2, L2, optimize=True)
        
        H2_valid = self.dH2_dxdx[self.valid_i, self.valid_j]  # shape: (num_pairs, hdim, hdim, hdim, hdim)
        
        V_alpha_beta = np.einsum('pa,pb,pabmn->mn',tmp1[self.valid_i], tmp2[self.valid_j], H2_valid, optimize=True)
        
        V_alpha_beta += V_alpha_beta.conj().T
        V_alpha_beta *= 0.5
           
        return V_alpha_beta

    def save_data(self):

            """
            Save data
                - couplings
            """

            with h5.File(self.save_file, 'a') as f:

                output = f['output']
                output.create_dataset('dH_dx', data=self.dH_dx)
                output.create_dataset('d2H_dxdx', data=self.dH2_dxdx)


            return

