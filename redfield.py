import numpy as np
from constants import hbar
import phonon
import math_func

class redfield:
   def __init__(self,V_alpha,V_alpha_beta,G_1ph,G_2ph,eigenvectors,eigenvalues,omega_q,n_alpha_q,Delta_alpha_q): 

      self.V_alpha = V_alpha
      self.V_alpha_beta = V_alpha_beta
      self.G_1ph = G_1ph
      self.G_2ph = G_2ph
      self.eigenvectors = eigenvectors
      self.eigenvalues = eigenvalues
      self.omega_q = omega_q
      self.n_alpha_q = n_alpha_q
      self.Delta_alpha_q = Delta_alpha_q

      self.hdim = self.V_alpha.shape[0]
      self.Nq = self.V_alpha.shape[2]

      self.omega_ij = np.zeros([self.hdim,self.hdim], dtype=np.float64)
      self.omega_ij = math_func.energy_diff(self.eigenvalues)

      self.R1_tensor = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
      self.R2_tensor = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)
      self.R4_tensor = np.zeros((self.hdim, self.hdim, self.hdim, self.hdim), dtype=np.complex128)

      self.init_R1()
      self.init_R2()
      self.init_R4()

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
                                 term_1 += (self.V_alpha[a,j,q] * self.V_alpha[j,c,q] * self.G_1ph[j,c,q])
                                             
                              # Second summation term
                              term_2 = self.V_alpha[a,c,q] * self.V_alpha[d,b,q] * self.G_1ph[a,c,q]
                              
                              # Third summation term
                              term_3 = self.V_alpha[a,c,q] * self.V_alpha[d,b,q] * self.G_1ph[b,d,q]
                              
                              # Forth summation term
                              term_4 = 0
                              for j in range(self.hdim):
                                 term_4 += (self.V_alpha[c,d,q] * self.V_alpha[a,j,q] * self.G_1ph[j,d,q])
                                             
                              
                              # Combine terms and update the tensor
                              self.R1_tensor[a, b, c, d] += (-np.pi / (2 * hbar) * (term_1 - term_2 - term_3 + term_4) )

      return

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
                                 
                                 # Compute terms
                                 term1 += (delta_bd * self.V_alpha_beta[a, j,alpha,beta] * self.V_alpha_beta[j, c,alpha,beta] * self.G_2ph[j, c, alpha, beta])
            
                              term2 = (-self.V_alpha_beta[a, c,alpha,beta] * self.V_alpha_beta[d, b,alpha,beta] * self.G_2ph[a, c, alpha, beta])
                              term3 = (-self.V_alpha_beta[a, c,alpha,beta] * self.V_alpha_beta[d, b,alpha,beta] * self.G_2ph[b, d, alpha, beta])
                              
                              for j in range(self.hdim):
                                 # Delta terms
                                 delta_ca = 1 if c == a else 0

                                 term4 += (delta_ca * self.V_alpha_beta[d, j,alpha,beta] * self.V_alpha_beta[j, b,alpha,beta] * self.G_2ph[j, d, alpha, beta])
                              
                              # Combine all terms
                              self.R2_tensor[a, b, c, d] += (-np.pi / (4 * hbar**2) * (term1 + term2 + term3 + term4))

      return

   def init_R4(self):

      # Compute matrix element <a|V_alpha|b>
      def mat(a, b, alpha):
         tmp = np.dot(self.V_alpha[:, :, alpha],self.eigenvectors[b])
         tmp1 = np.dot(self.eigenvectors[a], tmp)

         return tmp1

      


      for alpha in range(self.Nq):
         for beta in range(self.Nq):
            if alpha >= beta:
               continue
            if alpha == beta:
               delta_alpha_beta  = 1
            else:
               delta_alpha_beta = 0

            A_alpha_beta = 1-(0.75*delta_alpha_beta)
            B_alpha_beta = 1-(0.5*delta_alpha_beta)

            for a in range(self.hdim):
                  for b in range(self.hdim):
                      
                     W_mm = 0
                     W_pp = 0
                     W_pm = 0
                     W_mp = 0

                     for c in range(self.hdim):

                        term1 =  (mat(a, c, alpha)*mat(c, b, beta))/(self.eigenvalues[c]-self.eigenvalues[b]-self.omega_q[beta])
                        term2 =  (mat(a, c, beta)*mat(c, b, alpha))/(self.eigenvalues[c]-self.eigenvalues[b]-self.omega_q[alpha])

                        W_mm += np.abs(term1 + term2)**2

                        term3 =  (mat(a, c, alpha)*mat(c, b, beta))/(self.eigenvalues[c]-self.eigenvalues[b]+self.omega_q[beta])
                        term4 =  (mat(a, c, beta)*mat(c, b, alpha))/(self.eigenvalues[c]-self.eigenvalues[b]+self.omega_q[alpha])
                        W_pp += np.abs(term3 + term4)**2

                        term5 =  (mat(a, c, alpha)*mat(c, b, beta))/(self.eigenvalues[c]-self.eigenvalues[b]+self.omega_q[beta])
                        term6 =  (mat(a, c, beta)*mat(c, b, alpha))/(self.eigenvalues[c]-self.eigenvalues[b]-self.omega_q[alpha])

                        W_pm += np.abs(term5 + term6)**2

                        term7 =  (mat(a, c, alpha)*mat(c, b, beta))/(self.eigenvalues[c]-self.eigenvalues[b]-self.omega_q[beta])
                        term8 =  (mat(a, c, beta)*mat(c, b, alpha))/(self.eigenvalues[c]-self.eigenvalues[b]+self.omega_q[alpha])

                        W_mp += np.abs(term7 + term8)**2

                     W_mm = W_mm * self.n_alpha_q[alpha] * self.n_alpha_q[beta] * math_func.lorentzian(self.omega_ij[:,:] - self.omega_q[alpha] - self.omega_q[beta], self.Delta_alpha_q) 
                     W_pp = W_pp * (self.n_alpha_q[alpha]+1) * (self.n_alpha_q[beta]+1) * math_func.lorentzian(self.omega_ij[:,:] + self.omega_q[alpha] + self.omega_q[beta], self.Delta_alpha_q) 
                     W_pm = W_pm * self.n_alpha_q[alpha] * (self.n_alpha_q[beta]+1) * math_func.lorentzian(self.omega_ij[:,:] - self.omega_q[alpha] + self.omega_q[beta], self.Delta_alpha_q) 
                     W_mp = W_mp * (self.n_alpha_q[alpha]+1) * self.n_alpha_q[beta] * math_func.lorentzian(self.omega_ij[:,:] + self.omega_q[alpha] - self.omega_q[beta], self.Delta_alpha_q)

                     self.R4_tensor += (np.pi/(2*hbar**2))*(A_alpha_beta*W_mm + A_alpha_beta*W_pp + B_alpha_beta*W_pm + A_alpha_beta*W_mp) 
                     
                      
      return