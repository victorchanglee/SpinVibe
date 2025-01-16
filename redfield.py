import numpy as np
from constants import hbar

class redfield:
   def __init__(self,V_alpha_q,V_alpha_beta,G_1ph,G_2ph): 

      self.V_alpha_q = V_alpha_q
      self.V_alpha_beta = V_alpha_beta
      self.G_1ph = G_1ph
      self.G_2ph = G_2ph

      self.hdim = self.V_alpha_q.shape[0]
      self.Nq = self.V_alpha_q.shape[2]

      self.R1_tensor = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
      self.R2_tensor = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)

      self.init_R1()
      self.init_R2()

      return

   def init_R1(self):

      for a in range(self.hdim):
         for b in range(self.hdim):
            for c in range(self.hdim):
                  for d in range(self.hdim):
                     # Sum over alpha, q
                        for q in range(self.Nq):
                              # First summation term
                              term_1 = 0
                              for j in range(self.hdim):
                                 term_1 += (self.V_alpha_q[a,j,q] * self.V_alpha_q[j,c,q] * self.G_1ph[j,c,q])
                                             
                              # Second summation term
                              term_2 = self.V_alpha_q[a,c,q] * self.V_alpha_q[d,b,q] * self.G_1ph[a,c,q]
                              
                              # Third summation term
                              term_3 = self.V_alpha_q[a,c,q] * self.V_alpha_q[d,b,q] * self.G_1ph[b,d,q]
                              
                              # Forth summation term
                              term_4 = 0
                              for j in range(self.hdim):
                                 term_4 += (self.V_alpha_q[c,d,q] * self.V_alpha_q[a,j,q] * self.G_1ph[j,d,q])
                                             
                              
                              # Combine terms and update the tensor
                              self.R1_tensor[a, b, c, d] += (-np.pi / (2 * hbar) * (term_1 - term_2 - term_3 + term_4) )


   def init_R2(self):

      for alpha in range(self.Nq):
         for beta in range(self.Nq):
            if beta < alpha:
                  continue

            for a in range(self.hdim):
                  for b in range(self.hdim):
                     for c in range(self.hdim):
                        for d in range(self.hdim):
                              term1 = 0
                              term2 = 0
                              term3 = 0
                              term4 = 0
                              
                              for j in range(self.hdim):
                                 # Delta terms
                                 delta_bd = 1 if b == d else 0
                                 delta_ca = 1 if c == a else 0
                                 
                                 # Compute terms
                                 term1 += (
                                    delta_bd * V[alpha, q, a, j] * V[alpha - q, beta, j, c]
                                    * G2_ph(phonon_frequencies[j, c], phonon_frequencies[alpha, q], phonon_frequencies[beta, q_prime]))
                                 term2 += (
                                    -V[alpha, q, a, c] * V[alpha - q, beta, d, b]
                                    * G2_ph(energy_differences[a, c], phonon_frequencies[alpha, q], phonon_frequencies[beta, q_prime]))
                                 term3 += (-V[alpha, q, a, c] * V[alpha - q, beta, d, b]
                                    * G(energy_differences[b, d], phonon_frequencies[alpha, q], phonon_frequencies[beta, q_prime]))
                                 term4 += (delta_ca * V[alpha, q, d, j] * V[alpha - q, beta, j, b]
                                    * G(energy_differences[j, d], phonon_frequencies[alpha, q], phonon_frequencies[beta, q_prime]))
                              
                              # Combine all terms
                              R2_ph[a, b, c, d] += (-pi / (4 * hbar**2) * (term1 + term2 + term3 + term4))