import numpy as np
from constants import hbar
import math_func

def G_1ph(omega_ij, omega_alpha_q, Delta_alpha_q, T):
    """
    Compute G^{1-ph}(omega_ij, omega_alpha_q) based on the given equation.
    
    Parameters:
        omega_ij (float): Energy difference (ω_ij).
        omega_alpha_q (float): Phonon frequency (ω_{αq}).
        Delta_alpha_q (float): Broadening parameter (Δ_{αq}).
        n_alpha_q (float): Phonon occupation number (n̄_{αq}).
    
    Returns:
        float: Value of G^{1-ph}.
    """
    

    term1 = math_func.lorentzian(omega_ij - omega_alpha_q, Delta_alpha_q) 
    term2 = math_func.lorentzian(omega_ij + omega_alpha_q, Delta_alpha_q) 
        
    G_1ph_value =  (term1 * math_func.bose_einstein(omega_alpha_q,T) + term2 * (math_func.bose_einstein(omega_alpha_q,T) + 1))
                
    return G_1ph_value

                 
def R1_tensor(V_alpha,eigenvalues,omega_alpha,Delta_alpha_q, T):


   hdim = V_alpha.shape[2]
   Nq = V_alpha.shape[0]
   Nomega = V_alpha.shape[1]

   R1_tensor = np.zeros((hdim, hdim, hdim, hdim), dtype=np.complex128)
   prefactor = -np.pi / (2 * hbar**2)
   omega = eigenvalues[:, None] - eigenvalues[None, :]

   for a in range(hdim):
      for b in range(hdim):
         for c in range(hdim):
               for d in range(hdim):
                  for alpha in range(Nomega):
                     for q in range(Nq):
                        term1 = 0.0
                        if b == d:
                           for j in range(hdim):
                              V_aj = V_alpha[q,alpha, a, j]
                              V_jc = V_alpha[q,alpha, j, c]
                              G_term1 = G_1ph(omega[j, c], omega_alpha[q,alpha], Delta_alpha_q, T)
                              term1 += V_aj * V_jc * G_term1

                        V_ac = V_alpha[q,alpha, a, c]
                        V_db = V_alpha[q,alpha, d, b]
                        G_term2 = G_1ph(omega[b, d], omega_alpha[q,alpha], Delta_alpha_q, T)
                        term2 = -V_ac * V_db * G_term2


                        # Term 3: -V_{ac}^α V_{db}^α G(ω_ac, ω_α)
                        G_term3 = G_1ph(omega[a, c], omega_alpha[q,alpha], Delta_alpha_q, T)
                        term3 = -V_ac * V_db * G_term3

                        # Term 4: δ_{ca} * sum_j [V_{dj}^α V_{jb}^α G(ω_jd, ω_α)]
                        term4 = 0.0
                        if c == a:
                           for j in range(hdim):
                              V_dj = V_alpha[q,alpha, d, j]
                              V_jb = V_alpha[q,alpha, j, b]
                              G_term4 = G_1ph(omega[j, d], omega_alpha[q,alpha], Delta_alpha_q, T)
                              term4 += V_dj * V_jb * G_term4

                        # Combine terms
                        R1_tensor[a, b, c, d] += prefactor * (term1 + term2 + term3 + term4)


   return R1_tensor

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

def R4_tensor(V_alpha,omega_q, eigenvalues, eigenvectors, n_alpha_q, Delta_alpha_q):

   hdim = V_alpha.shape[2]
   Nq = V_alpha.shape[0]
   Nomega = V_alpha.shape[1]

   omega_ij = np.zeros([hdim,hdim], dtype=np.float64)
   omega_ij = math_func.energy_diff(eigenvalues)

   R4_tensor = np.zeros((hdim, hdim, hdim, hdim), dtype=np.complex128)  

   for omega in range(Nomega):
      for alpha in range(Nq):
         for beta in range(Nq):
            if alpha >= beta:
               continue
            if alpha == beta:
               delta_alpha_beta  = 1
            else:
               delta_alpha_beta = 0

            A_alpha_beta = 1-(0.75*delta_alpha_beta)
            B_alpha_beta = 1-(0.5*delta_alpha_beta)

            for a in range(hdim):
                  for b in range(hdim):
                        
                     W_mm = 0
                     W_pp = 0
                     W_pm = 0
                     W_mp = 0

                     for c in range(hdim):

                        term1 =  (math_func.mat(a, c, alpha,omega,eigenvectors,V_alpha)*math_func.mat(c, b, beta,omega,eigenvectors,V_alpha))/(eigenvalues[c]-eigenvalues[b]-omega_q[beta,omega])
                        term2 =  (math_func.mat(a, c, beta,omega,eigenvectors,V_alpha)*math_func.mat(c, b, alpha,omega,eigenvectors,V_alpha))/(eigenvalues[c]-eigenvalues[b]-omega_q[alpha,omega])

                        W_mm += np.abs(term1 + term2)**2

                        term3 =  (math_func.mat(a, c, alpha,omega,eigenvectors,V_alpha)*math_func.mat(c, b, beta,omega,eigenvectors,V_alpha))/(eigenvalues[c]-eigenvalues[b]+omega_q[beta,omega])
                        term4 =  (math_func.mat(a, c, beta,omega,eigenvectors,V_alpha)*math_func.mat(c, b, alpha,omega,eigenvectors,V_alpha))/(eigenvalues[c]-eigenvalues[b]+omega_q[alpha,omega])
                        W_pp += np.abs(term3 + term4)**2

                        term5 =  (math_func.mat(a, c, alpha,omega,eigenvectors,V_alpha)*math_func.mat(c, b, beta,omega,eigenvectors,V_alpha))/(eigenvalues[c]-eigenvalues[b]+omega_q[beta,omega])
                        term6 =  (math_func.mat(a, c, beta,omega,eigenvectors,V_alpha)*math_func.mat(c, b, alpha,omega,eigenvectors,V_alpha))/(eigenvalues[c]-eigenvalues[b]-omega_q[alpha,omega])

                        W_pm += np.abs(term5 + term6)**2

                        term7 =  (math_func.mat(a, c, alpha,omega,eigenvectors,V_alpha)*math_func.mat(c, b, beta,omega,eigenvectors,V_alpha))/(eigenvalues[c]-eigenvalues[b]-omega_q[beta,omega])
                        term8 =  (math_func.mat(a, c, beta,omega,eigenvectors,V_alpha)*math_func.mat(c, b, alpha,omega,eigenvectors,V_alpha))/(eigenvalues[c]-eigenvalues[b]+omega_q[alpha,omega])

                        W_mp += np.abs(term7 + term8)**2

                     W_mm = W_mm * n_alpha_q[alpha,omega] * n_alpha_q[beta,omega] * math_func.lorentzian(omega_ij[:,:] - omega_q[alpha,omega] - omega_q[beta,omega], Delta_alpha_q) 
                     W_pp = W_pp * (n_alpha_q[alpha,omega]+1) * (n_alpha_q[beta,omega]+1) * math_func.lorentzian(omega_ij[:,:] + omega_q[alpha,omega] + omega_q[beta,omega], Delta_alpha_q) 
                     W_pm = W_pm * n_alpha_q[alpha,omega] * (n_alpha_q[beta,omega]+1) * math_func.lorentzian(omega_ij[:,:] - omega_q[alpha,omega] + omega_q[beta,omega], Delta_alpha_q) 
                     W_mp = W_mp * (n_alpha_q[alpha,omega]+1) * n_alpha_q[beta,omega] * math_func.lorentzian(omega_ij[:,:] + omega_q[alpha,omega] - omega_q[beta,omega], Delta_alpha_q)

                     R4_tensor += (np.pi/(2*hbar**2))*(A_alpha_beta*W_mm + A_alpha_beta*W_pp + B_alpha_beta*W_pm + A_alpha_beta*W_mp) 
                     
                     
   return R4_tensor
