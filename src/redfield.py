import numpy as np
from constants import hbar, c, hbar_SI, cm2J
import math_func
import coupling
from mpi4py import MPI
from matplotlib import pyplot as plt

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
      omega_alpha_q = omega_alpha_q 
      omega_ij =  omega_ij 
      Delta_alpha_q = self.Delta_alpha_q 

      delta = math_func.lorentzian((omega_ij - omega_alpha_q), Delta_alpha_q)
      n_aq = math_func.bose_einstein(omega_alpha_q, self.T)
      G_1ph = delta * n_aq + delta * (n_aq + 1)

      #term1_numerator = Delta_alpha_q
      #term1_denominator = Delta_alpha_q**2 + (omega_ij - omega_alpha_q)**2
      #n_1 = math_func.bose_einstein(omega_alpha_q, self.T)
      #term1 = (term1_numerator / term1_denominator) * n_1
      
      #term2_numerator = Delta_alpha_q
      #term2_denominator = Delta_alpha_q**2 + (omega_ij + omega_alpha_q)**2
      #n_2 = math_func.bose_einstein(omega_alpha_q, self.T)
      #term2 = (term2_numerator / term2_denominator) * (n_2 + 1)
      
      #G_1ph = (term1 + term2) 
      G_1ph = G_1ph / (2* np.pi* c)           
      return G_1ph

   def G_2ph(self, omega_ij, omega_alpha_q, omega_beta_qp):

      #cm-1 to rad/s conversion
      omega_alpha_q = omega_alpha_q
      omega_beta_qp = omega_beta_qp 
      omega_ij = omega_ij 
      Delta = self.Delta_alpha_q

      Delta = self.Delta_alpha_q + self.Delta_alpha_q
      
      n_alpha = math_func.bose_einstein(omega_alpha_q, self.T)
      n_beta = math_func.bose_einstein(omega_beta_qp, self.T)

      # Four terms of the Green's function
      term1 = math_func.lorentzian(omega_ij - omega_alpha_q - omega_beta_qp, Delta) * n_alpha * n_beta
      term2 = math_func.lorentzian(omega_ij + omega_alpha_q + omega_beta_qp, Delta) * (n_alpha + 1) * (n_beta + 1)
      term3 = math_func.lorentzian(omega_ij + omega_alpha_q - omega_beta_qp, Delta) * (n_alpha + 1) * n_beta
      term4 = math_func.lorentzian(omega_ij - omega_alpha_q + omega_beta_qp, Delta) * n_alpha * (n_beta + 1)

      #term1 = (Delta / (Delta**2 + (omega_ij - omega_alpha_q - omega_beta_qp)**2)) * n_alpha * n_beta
      #term2 = (Delta / (Delta**2 + (omega_ij + omega_alpha_q + omega_beta_qp)**2)) * (n_alpha + 1) * (n_beta + 1)
      #term3 = (Delta / (Delta**2 + (omega_ij - omega_alpha_q + omega_beta_qp)**2)) * n_alpha * (n_beta + 1)
      #term4 = (Delta / (Delta**2 + (omega_ij + omega_alpha_q - omega_beta_qp)**2)) * (n_alpha + 1) * n_beta

      G2ph = (term1 + term2 + term3 + term4) / (2*np.pi*c)


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
                        
                           #print(term1, G_term1)
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
      prefactor = -np.pi / (4 * hbar_SI**2) 

      
      
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
         
         for a in range(self.hdim):
            for b in range(self.hdim):
                  for c in range(self.hdim):
                     for d in range(self.hdim):
                        term_sum = 0.0
                        
                        # First term: δ_bd * Σ_j V^{αβ}_{aj} V^{αβ}_{jc} G^{2-ph}(ω_{jc}, ω_α, ω_β)
                        if b == d:  # Kronecker delta δ_bd
                              for j in range(self.hdim):
                                 omega_jc = self.omega[j, c]
                                 G_jc = G_cache[omega_jc]
                                 term_sum += V_alpha_beta[a, j] * V_alpha_beta[j, c] * G_jc
                        
                        # Second term: -V^{αβ}_{ac} V^{αβ}_{db} G^{2-ph}(ω_{bd}, ω_α, ω_β)
                        omega_bd = self.omega[b, d]
                        G_bd = G_cache[omega_bd]
                        term_sum -= V_alpha_beta[a, c] * V_alpha_beta[d, b] * G_bd
                        
                        # Third term: -V^{αβ}_{ac} V^{αβ}_{db} G^{2-ph}(ω_{ac}, ω_α, ω_β)
                        omega_ac = self.omega[a, c]
                        G_ac = G_cache[omega_ac]
                        term_sum -= V_alpha_beta[a, c] * V_alpha_beta[d, b] * G_ac
                        
                        # Fourth term: δ_ca * Σ_j V^{αβ}_{dj} V^{αβ}_{jb} G^{2-ph}(ω_{jd}, ω_α, ω_β)
                        if c == a:  # Kronecker delta δ_ca
                              for j in range(self.hdim):
                                 omega_jd = self.omega[j, d]  # Note: corrected from ω_{jb} to ω_{jd}
                                 G_jd = G_cache[omega_jd]
                                 term_sum += V_alpha_beta[d, j] * V_alpha_beta[j, b] * G_jd
                        
                        # Apply prefactor and add to result
                        R2_local[a, b, c, d] += prefactor * term_sum

      # Gather results from all processes
      R2_tensor = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
      comm.Allreduce(R2_local, R2_tensor, op=MPI.SUM)
      
      return R2_tensor
