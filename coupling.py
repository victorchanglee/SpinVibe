import numpy as np
from constants import hbar_SI, c
import hamiltonian
import math_func
from mpi4py import MPI

class coupling:
    def __init__(self, B, S, T, eigenvectors, q_vector, omega_q, R_vectors, L_vectors,rot_mat,Ncells, file_reader):
    
        self.file_reader = file_reader
        self.B = B # Magnetic field in cm-1
        self.S = S # Spin
        self.T = T # Temperatur in K
        self.Ns = int(2*self.S + 1) # Number of spins
        self.hdim = self.Ns
        self.eigenvectors = eigenvectors # Eigenvectors of the spin Hamiltonian
        self.indices = self.file_reader.read_indices() # Mapping of the molecule indices in the crystal
        self.N = len(self.indices) # Number of atoms in the molecule
        self.masses = self.file_reader.read_mol_masses() # Atomic mass of the molecule
        self.q_vector = q_vector # Phonon mode index and wave vector q 
        self.omega_q = omega_q # Phonon frequencies in cm-1
        self.Nq = L_vectors.shape[0] # Number of q points
        self.Nomega = L_vectors.shape[1] # Number of phonons at q
        self.Ncells = Ncells
        self.L_vectors = L_vectors # Phonon eigenvectors
        self.R_vectors = R_vectors # Atomic position vectors
        self.R_vectors_mol = R_vectors[self.indices,:] # Position vectors of the molecule in the crystal
        self.L_vectors_mol = np.zeros((self.Nq, self.Nomega, self.N, 3), dtype=np.complex128) # Eigenvectors of the molecule in the crystal
        self.L_vectors_mol = L_vectors[:,:,self.indices,:] 
        self.N_atoms = R_vectors.shape[0]  # Number of atoms in the crystal
        self.rot_mat = rot_mat # Rotational matrix for hte molecule to match the coordinates in the crystal
        
        
        d_tensor, g_tensor = self.file_reader.read_orca() # g-matrix and ZFS D-tensor

        #Rotate the molecular cartesian matrices to match the crystal coordinates
        self.g_tensor = self.rot_mat @ g_tensor @ self.rot_mat.T
        self.d_tensor = self.rot_mat @ d_tensor @ self.rot_mat.T

        self.dH_dx = np.zeros((self.N,3,self.hdim,self.hdim), dtype=np.complex128)

        #Read linear displacement g-matrix and ZFS D-tensor
        D_d1, G_d1,self.disp1 = self.file_reader.read_d1()
        self.G_d1 = self.rot_mat @ G_d1 @ self.rot_mat
        self.D_d1 = self.rot_mat @ D_d1 @ self.rot_mat
        self.compute_dH_dx() #Compute dH/dx

        #Read quadratic displacement g-matrix and ZFS D-tensor
        D_d2, G_d2, self.disp2 = self.file_reader.read_d2()
        self.G_d2 = self.rot_mat @ G_d2 @ self.rot_mat
        self.D_d2 = self.rot_mat @ D_d2 @ self.rot_mat

        self.dH2_dxdx = np.zeros((self.N,self.N,3,3,self.hdim,self.hdim), dtype=np.complex128)
        self.compute_d2H_dxdx() #Compute d2H/dxdx'

        #Compute linear spin-phonon coupling
        self.V_alpha = np.zeros((self.Nq, self.Nomega,self.hdim, self.hdim), dtype=np.complex128)
        self.V_alpha = self.compute_V_alpha_q()

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

    def compute_V_alpha_q(self):
        """
        Compute the interaction matrix elements V^{alpha q}_{aj}.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Create a flattened list of (q, omega) pairs
        all_q_omega_pairs = []
        for q_idx in range(self.Nq):
            for omega_idx in range(self.Nomega):
                all_q_omega_pairs.append((q_idx, omega_idx))
        
        # Distribute the (q, omega) pairs across processes
        local_q_omega_pairs = np.array_split(np.array(all_q_omega_pairs, dtype=object), size)[rank]
        N_local_pairs = len(local_q_omega_pairs)

        if rank == 0:
            if size > 1:
                print("Parallelizing computation across processes...")
                print(f"Number of processes: {size}")
                print(f"Number of task: {len(all_q_omega_pairs)}")
                print(f"Number of tasks per process: {N_local_pairs}")
                print("\n")
            else:
                print("Running in serial mode.")
                print("\n")

        # Initialize local temporary array
        tmp_local = np.zeros((N_local_pairs, self.hdim, self.hdim), dtype=np.complex128)

        term = np.einsum('ijkl,klab -> ijkab', self.L_vectors_mol, self.dH_dx)  # shape: (Nq, Nomega, N, hdim, hdim)

        # Fill tmp_local with local (q, omega) computations
        for local_idx, (q, omega) in enumerate(local_q_omega_pairs):
            tmp1_part = np.zeros(self.N, dtype=np.complex128)
            for atom in range(self.N):
                freq = 2 * np.pi * self.omega_q[q, omega] * c # Convert to radian/s
                # Handle cases where freq might be zero or very small to avoid division by zero
                if freq == 0:
                    prefactor = 0.0 # Or some other appropriate handling
                else:
                    prefactor = np.sqrt(hbar_SI / (self.Nq * freq * self.masses[atom]))
                    prefactor *= 1 / np.sqrt(self.Ncells)
                    prefactor *= 1E10  # Convert to A units
                    
                q_dot_R = np.dot(self.q_vector[q], self.R_vectors_mol[atom])
                exponential = np.exp(1j * q_dot_R)
                tmp1_part[atom] = prefactor * exponential

            # Slice 'term' for the current q and omega
            current_term_slice = term[q, omega]  # shape: (N, hdim, hdim)

            # Perform the einsum for this (q, omega) pair
            tmp_local[local_idx] = np.einsum('k,kab->ab', tmp1_part, current_term_slice)


        # Flatten local array for Gatherv
        tmp_local_flat = tmp_local.ravel()

        # Calculate sendcounts and displs for Gatherv
        # Each process sends N_local_pairs * hdim * hdim elements
        sendcounts = np.array([len(chunk) * self.hdim * self.hdim for chunk in np.array_split(np.array(all_q_omega_pairs, dtype=object), size)])
        displs = np.insert(np.cumsum(sendcounts), 0, 0)[0:-1]

        recvbuf = None
        if rank == 0:
            recvbuf = np.empty(self.Nq * self.Nomega * self.hdim * self.hdim, dtype=np.complex128)

        # Gather all data to root
        comm.Gatherv(sendbuf=tmp_local_flat, recvbuf=(recvbuf, sendcounts, displs, MPI.COMPLEX16), root=0)

        V_alpha = None
        if rank == 0:
            tmp_full = recvbuf.reshape((self.Nq, self.Nomega, self.hdim, self.hdim))
            H_herm = np.conj(np.transpose(tmp_full, axes=(0, 1, 3, 2)))
            V_alpha = 0.5 * (tmp_full + H_herm)
        else:
            V_alpha = np.empty((self.Nq, self.Nomega, self.hdim, self.hdim), dtype=np.complex128)

        # Broadcast result to all processes
        comm.Bcast(V_alpha, root=0)

        return V_alpha

    def pre_compute_V_alpha_beta_q(self):
        
        self.w = 2 * np.pi * self.omega_q * c

        base_factor = np.sqrt(hbar_SI / (self.Nq )) 
    
        self.prefactor = np.zeros((self.Nq, self.Nomega, self.N))
        
        for nq in range(self.Nq):
            for nomega in range(self.Nomega):
                w_val = self.w[nq, nomega]
                # Each atom has its own prefactor
                self.prefactor[nq, nomega, :] = base_factor / np.sqrt(w_val * self.masses) 
                self.prefactor[nq, nomega, :] *= 1 / np.sqrt(self.Ncells)
                self.prefactor[nq, nomega, :] *= 1E10  # Convert to A units
        
        self.exp = np.zeros((self.Nq, self.N), dtype=np.complex128)
        
        for nq in range(self.Nq):
            # R_vectors_mol @ q_vector[nq] gives dot product for each atom
            # Shape: (N,) = (N, 3) @ (3,)
            self.exp[nq, :] = np.exp(1j * self.R_vectors_mol @ self.q_vector[nq])

        #Ignore when atom1 = atom2 
        i_indices = np.arange(self.N)
        self.valid_i = np.repeat(i_indices, self.N-1)
        self.valid_j = np.concatenate([np.arange(i) for i in range(self.N)] + 
                                [np.arange(i+1, self.N) for i in range(self.N)])


    def compute_V_alpha_beta_q(self, Nq1, Nq2, Nomega1, Nomega2):

        prefactor1 = self.prefactor[Nq1, Nomega1, :]  
        prefactor2 = self.prefactor[Nq2, Nomega2, :]  
        exp1 = self.exp[Nq1, :]  
        exp2 = self.exp[Nq2, :]  
        L1 = self.L_vectors_mol[Nq1, Nomega1, :, :] 
        L2 = self.L_vectors_mol[Nq2, Nomega2, :, :]  
    
        combined_factors = (prefactor1[self.valid_i] * prefactor2[self.valid_j] * 
                           exp1[self.valid_i] * exp2[self.valid_j])
        
        L1_valid = L1[self.valid_i]  # shape: (num_pairs, hdim)
        L2_valid = L2[self.valid_j]  # shape: (num_pairs, hdim)
        
        H2_valid = self.dH2_dxdx[self.valid_i, self.valid_j]  # shape: (num_pairs, hdim, hdim, hdim, hdim)
        
        V_alpha_beta = np.einsum('p,pa,pb,pabmn->mn', 
                                  combined_factors, L1_valid, L2_valid, H2_valid,
                                  optimize=True)
        
        V_alpha_beta += V_alpha_beta.conj().T
        V_alpha_beta *= 0.5
           
        return V_alpha_beta



