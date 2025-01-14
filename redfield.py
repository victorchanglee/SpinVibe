import numpy as np
from constants import hbar

class redfield:
   def __init__(self,V_alpha_q,G_1ph): 

      self.V_alpha_q = V_alpha_q
      self.G_1ph = G_1ph

      self.hdim = self.V_alpha_q.shape[0]
      self.Nq = self.V_alpha_q.shape[2]

      self.R_tensor = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)

      self.init_redfield()

      return

   def init_redfield(self):

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
                              self.R_tensor[a, b, c, d] += (-np.pi / (2 * hbar) * (term_1 - term_2 - term_3 + term_4) )

