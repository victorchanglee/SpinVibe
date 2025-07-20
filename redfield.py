import numpy as np
from constants import hbar, c, hbar_SI, cm2J
import math_func
import coupling
from mpi4py import MPI
from matplotlib import pyplot as plt
from tqdm import tqdm

class Redfield:

   def __init__(self, S, T, eigenvectors,eigenvalues,q_vector, omega_q, Delta_alpha_q, L_vectors):

      self.S = S
      self.T = T
      self.Ns = int(2*self.S + 1)
      self.hdim = self.Ns
      self.eigenvectors = eigenvectors # Eigenvectors of the spin Hamiltonian
      self.eigenvalues = eigenvalues
      self.q_vector = q_vector # Phonon mode index and wave vector q 
      self.omega_q = omega_q
      self.Delta_alpha_q = Delta_alpha_q # Broadening parameter
      self.Nq = L_vectors.shape[0] # Number of q points
      self.Nomega = L_vectors.shape[1] # Number of phonons at q

      self.omega = self.eigenvalues[:, None] - self.eigenvalues[None, :]

   def G_1ph(self,omega_ij, omega_alpha_q):
      """

      Parameters:
         omega_ij (float): Energy difference (ω_ij).
         omega_alpha_q (float): Phonon frequency (ω_{αq}).
         Delta_alpha_q (float): Broadening parameter (Δ_{αq}).
         n_alpha_q (float): Phonon occupation number (n̄_{αq}).
      
      Returns:
         float: Value of G^{1-ph}.
      """

      #cm-1 to rad/s conversion
      omega_alpha_q = 2 * np.pi * omega_alpha_q * c
      omega_ij = 2 * np.pi * omega_ij * c
      Delta_alpha_q = 2 * np.pi * self.Delta_alpha_q * c

      #delta = math_func.lorentzian((omega_ij - omega_alpha_q), Delta_alpha_q)
      #n_aq = math_func.bose_einstein(omega_alpha_q, self.T)
      #G_1ph = delta * n_aq + delta * (n_aq + 1)

      term1_numerator = Delta_alpha_q
      term1_denominator = Delta_alpha_q**2 + (omega_ij - omega_alpha_q)**2
      n_1 = math_func.bose_einstein(omega_alpha_q, self.T)
      term1 = (term1_numerator / term1_denominator) * n_1
      
      term2_numerator = Delta_alpha_q
      term2_denominator = Delta_alpha_q**2 + (omega_ij + omega_alpha_q)**2
      n_2 = math_func.bose_einstein(omega_alpha_q, self.T)
      term2 = (term2_numerator / term2_denominator) * (n_2 + 1)
      
      G_1ph = (term1 + term2) / np.pi           
      return G_1ph

   def G_2ph(self, omega_ij, omega_alpha_q, omega_beta_qp):

      #cm-1 to rad/s conversion
      omega_alpha_q = 2 * np.pi * omega_alpha_q * c
      omega_beta_qp = 2 * np.pi * omega_beta_qp * c
      omega_ij = 2 * np.pi * omega_ij * c
      Delta = 2 * np.pi * self.Delta_alpha_q * c

      Delta = self.Delta_alpha_q + self.Delta_alpha_q
      
      n_alpha = math_func.bose_einstein(omega_alpha_q, self.T)
      n_beta = math_func.bose_einstein(omega_beta_qp, self.T)

      term1 = (Delta / (Delta**2 + (omega_ij - omega_alpha_q - omega_beta_qp)**2)) * n_alpha * n_beta
      term2 = (Delta / (Delta**2 + (omega_ij + omega_alpha_q + omega_beta_qp)**2)) * (n_alpha + 1) * (n_beta + 1)
      term3 = (Delta / (Delta**2 + (omega_ij - omega_alpha_q + omega_beta_qp)**2)) * n_alpha * (n_beta + 1)
      term4 = (Delta / (Delta**2 + (omega_ij + omega_alpha_q - omega_beta_qp)**2)) * (n_alpha + 1) * n_beta

      G2ph = (term1 + term2 + term3 + term4) / np.pi
      return G2ph

   def R1_tensor(self, V_alpha):
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      size = comm.Get_size()

      omega_alpha = self.omega_q
      V_alpha = V_alpha * cm2J

      prefactor = -np.pi / (2 * hbar_SI**2)

      # Create all (q, alpha) tasks for parallelization
      all_tasks = []
      for q in range(self.Nq):
         for alpha in range(self.Nomega):
               all_tasks.append((q, alpha))

      # Distribute tasks across processes
      tasks = np.array_split(all_tasks, size)[rank]

      # Initialize local contribution
      R1_local = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)

      # Process assigned (q, alpha) pairs
      for q, alpha in tasks:
         # Initialize contribution for this (q, alpha) pair
         R1_qa = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
         
         # Loop over tensor indices
         for a in range(self.hdim):
               for b in range(self.hdim):
                  for c in range(self.hdim):
                     for d in range(self.hdim):
                           
                           # Term 1: δ_{bd} Σ_j V^α_{aj} V^α_{jc} G^{1-ph}(ω_{jc}, ω_α)
                           term1 = 0.0
                           if b == d:
                              for j in range(self.hdim):
                                 V_aj = V_alpha[q, alpha, a, j]
                                 V_jc = V_alpha[q, alpha, j, c]
                                 G_term1 = self.G_1ph(self.omega[j, c], omega_alpha[q, alpha])
                                 term1 += V_aj * V_jc * G_term1

                           # Term 2: V^α_{ac} V^α_{db} G^{1-ph}(ω_{bd}, ω_α)
                           V_ac = V_alpha[q, alpha, a, c]
                           V_db = V_alpha[q, alpha, d, b]
                           G_term2 = self.G_1ph(self.omega[b, d], omega_alpha[q, alpha])
                           term2 = V_ac * V_db * G_term2

                           # Term 3: V^α_{ac} V^α_{db} G^{1-ph}(ω_{ac}, ω_α)
                           G_term3 = self.G_1ph(self.omega[a, c], omega_alpha[q, alpha])
                           term3 = V_ac * V_db * G_term3

                           # Term 4: δ_{ca} Σ_j V^α_{dj} V^α_{jb} G^{1-ph}(ω_{jd}, ω_α)
                           term4 = 0.0
                           if c == a:
                              for j in range(self.hdim):
                                 V_dj = V_alpha[q, alpha, d, j]
                                 V_jb = V_alpha[q, alpha, j, b]
                                 G_term4 = self.G_1ph(self.omega[j, d], omega_alpha[q, alpha])
                                 term4 += V_dj * V_jb * G_term4

                           # Compute contribution for this (q, alpha, a, b, c, d)
                           R1_qa[a, b, c, d] = term1 - term2 - term3 + term4

         # Add this (q, alpha) contribution to local sum
         R1_local += R1_qa

      # Apply prefactor to local contribution
      R1_local *= prefactor

      # Gather and sum contributions from all processes
      R1_tensor = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
      comm.Allreduce(R1_local, R1_tensor, op=MPI.SUM)

      return R1_tensor
   
   def R2_tensor(self, init_Vq):
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      size = comm.Get_size()
      
      omega_alpha = self.omega_q
      prefactor = -np.pi / (4 * hbar_SI**2)  # Corrected to match equation
      
      # Build full list of tasks 
      all_tasks = []
      task_counter = 0
      for q1 in range(self.Nq):
         for alpha in range(self.Nomega):
               for beta in range(alpha + 1):
                  max_q2 = self.Nq if beta < alpha else q1 + 1
                  for q2 in range(max_q2):
                     # Only add task if it belongs to this process
                     if task_counter % size == rank:
                           all_tasks.append((q1, q2, alpha, beta))
                     task_counter += 1

      if rank == 0:
         print(f"Number of tasks per process: {len(all_tasks)}")

      R2_local = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)

      # Setup progress bar only for rank 0
      if rank == 0:
         pbar = tqdm(total=len(all_tasks), desc="Computing R2 tensor", unit="tasks", ncols=100)

      for q1, q2, alpha, beta in all_tasks:
         # Get V matrix for this (alpha, beta) pair
         V_alpha_beta = init_Vq.compute_V_alpha_beta_q(q1, q2, alpha, beta)
         V_alpha_beta = V_alpha_beta * cm2J
         
         # Get the frequencies omega_alpha and omega_beta
         omega_a = omega_alpha[q1, alpha]  # ω_α in the equation
         omega_b = omega_alpha[q2, beta]   # ω_β in the equation
         
         # Pre-compute all Green's functions we'll need
         G_cache = {}
         
         # Cache G functions for all unique frequencies
         unique_freqs = set()
         for i in range(self.hdim):
               for j in range(self.hdim):
                  unique_freqs.add(self.omega[i, j])
         
         for freq in unique_freqs:
               G_cache[freq] = self.G_2ph(freq, omega_a, omega_b)
         
         # Vectorized computation of terms
         for a in range(self.hdim):
               for c in range(self.hdim):
                  # Get all G values for this a,c pair
                  G_ac = G_cache[self.omega[a, c]]
                  G_jc_vals = np.array([G_cache[self.omega[j, c]] for j in range(self.hdim)])
                  
                  # Term 1: δ_bd * Σ_j V^α_aj V^α_jc G^{2-ph}(ω_jc, ω_α, ω_β)
                  # Only contributes when b=d, so we can vectorize over b
                  term1_vals = np.sum(V_alpha_beta[a, :] * V_alpha_beta[:, c] * G_jc_vals)
                  for b in range(self.hdim):
                     R2_local[a, b, c, b] += term1_vals  # b=d case
                  
                  # Terms 2&3: -V^α_ac V^α_db G^{2-ph}(ω_bd/ω_ac, ω_α, ω_β)
                  V_ac_val = V_alpha_beta[a, c]
                  for b in range(self.hdim):
                     for d in range(self.hdim):
                           V_db_val = V_alpha_beta[d, b]
                           product = V_ac_val * V_db_val
                           
                           # Term 2: G(ω_bd)
                           G_bd = G_cache[self.omega[b, d]]
                           R2_local[a, b, c, d] -= product * G_bd
                           
                           # Term 3: G(ω_ac)
                           R2_local[a, b, c, d] -= product * G_ac
                  
                  # Term 4: δ_ca * Σ_j V^α_dj V^α_jb G^{2-ph}(ω_jb, ω_α, ω_β)
                  # Only contributes when c=a
                  if c == a:
                     for b in range(self.hdim):
                           G_jb_vals = np.array([G_cache[self.omega[j, b]] for j in range(self.hdim)])
                           for d in range(self.hdim):
                              term4_val = np.sum(V_alpha_beta[d, :] * V_alpha_beta[:, b] * G_jb_vals)
                              R2_local[a, b, c, d] += term4_val
            # Update progress bar only for rank 0
         if rank == 0:
               pbar.update(1)

      # Close progress bar for rank 0
      if rank == 0:
         pbar.close()

      # Gather results from all processes
      R2_tensor = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
      comm.Allreduce(R2_local, R2_tensor, op=MPI.SUM)

      # Apply prefactor
      R2_tensor = prefactor * R2_tensor
      
      return R2_tensor

   def R4_tensor(self, V_alpha):
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      size = comm.Get_size()

      hdim = V_alpha.shape[2]
      Nq = V_alpha.shape[0]
      Nomega = V_alpha.shape[1]

      all_tasks = []
      for q in range(self.Nq):
         for alpha in range(self.Nomega):
               all_tasks.append((q, alpha))

      all_tasks = np.array_split(all_tasks, size)[rank]

      # Precompute matrix elements M[q, alpha]
      M = np.zeros((Nq, Nomega, hdim, hdim), dtype=np.complex128)
      for q, alpha in all_tasks:
         V = V_alpha[q, alpha]
         M[q, alpha] = self.eigenvectors.conj().T @ V @ self.eigenvectors

      # Energy differences and tensor initialization
      omega_ij = math_func.energy_diff(self.eigenvalues)
      R4_local = np.zeros((hdim, hdim, hdim, hdim), dtype=np.complex128)

      for q, alpha in all_tasks:
         for beta in range(Nomega):
            if alpha >= beta:
               continue
            delta_alpha_beta = 1 if alpha == beta else 0

            M_alpha = M[q, alpha]
            M_beta = M[q, beta]

            omega_q_alpha = self.omega_q[q, alpha]
            omega_q_beta = self.omega_q[q, beta]

            # Compute all W terms
            term1 = M_alpha @ (M_beta.T / (omega_ij - omega_q_beta))
            term2 = M_beta @ (M_alpha.T / (omega_ij - omega_q_alpha))
            W_mm = np.abs(term1 + term2)**2

            term3 = M_alpha @ (M_beta.T / (omega_ij + omega_q_beta))
            term4 = M_beta @ (M_alpha.T / (omega_ij + omega_q_alpha))
            W_pp = np.abs(term3 + term4)**2

            term5 = M_alpha @ (M_beta.T / (omega_ij + omega_q_beta))
            term6 = M_beta @ (M_alpha.T / (omega_ij - omega_q_alpha))
            W_pm = np.abs(term5 + term6)**2

            term7 = M_alpha @ (M_beta.T / (omega_ij - omega_q_beta))
            term8 = M_beta @ (M_alpha.T / (omega_ij + omega_q_alpha))
            W_mp = np.abs(term7 + term8)**2

            # Bose factors
            n_alpha = math_func.bose_einstein(omega_q_alpha, self.T)
            n_beta  = math_func.bose_einstein(omega_q_beta, self.T)

            bose_mm = (n_alpha + 1) * n_beta
            bose_pp = (n_alpha + 1) * (n_beta + 1)
            bose_pm = n_alpha * n_beta
            bose_mp = (n_alpha + 1) * n_beta

            # Lorentzian factors
            lorentz_mm = math_func.lorentzian(omega_ij - omega_q_alpha - omega_q_beta, self.Delta_alpha_q)
            lorentz_pp = math_func.lorentzian(omega_ij + omega_q_alpha + omega_q_beta, self.Delta_alpha_q)
            lorentz_pm = math_func.lorentzian(omega_ij - omega_q_alpha + omega_q_beta, self.Delta_alpha_q)
            lorentz_mp = math_func.lorentzian(omega_ij + omega_q_alpha - omega_q_beta, self.Delta_alpha_q)

            # Final W's
            W_mm *= bose_mm * lorentz_mm
            W_pp *= bose_pp * lorentz_pp
            W_pm *= bose_pm * lorentz_pm
            W_mp *= bose_mp * lorentz_mp

            A = 1 - (0.75 * delta_alpha_beta)
            B = 1 - (0.5 * delta_alpha_beta)

            R4_local += (np.pi / (2 * hbar)) * (A * W_mm + A * W_pp + B * W_pm + A * W_mp)

      # Combine results from all ranks
      R4_total = np.zeros_like(R4_local)
      comm.Allreduce(R4_local, R4_total, op=MPI.SUM)

      return R4_total