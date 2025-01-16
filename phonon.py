import numpy as np
from constants import hbar,k_B
import math_func

class phonon:
    def __init__(self,omega_ij, omega_alpha_q, Delta_alpha_q, T):
        
        self.T = T

        self.omega_ij = omega_ij
        self.omega_alpha_q = omega_alpha_q
        self.Delta_alpha_q = Delta_alpha_q
        
        self.hdim = self.omega_ij.shape[0]
        self.Nq = len(self.omega_alpha_q)

        self.n_alpha_q = np.zeros([self.Nq],dtype=np.float64)
        
        self.bose_einstein()

        self.G_1ph_value = np.zeros([self.hdim,self.hdim,self.Nq],dtype=np.float64)

        self.G_1ph()

        self.G_2ph_value = np.zeros([self.hdim,self.hdim,self.Nq,self.Nq],dtype=np.float64)

        self.G_2ph()

        #self.G_2ph()

        return 
    
    def bose_einstein(self):
        """
        Compute the Bose-Einstein occupation number.
        
        Parameters:
            omega_alpha_q (float): Phonon frequency (ω_{αq}).
            T (float): Temperature in Kelvin.
        
        Returns:
            float: Bose-Einstein occupation number (n̄_{αq}).
        """

        for q in range(self.Nq):
            self.n_alpha_q[q] = 1 / (np.exp(self.omega_alpha_q[q] / (k_B * self.T)) - 1)

        return self.n_alpha_q
    
    def G_1ph(self):
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
        for q in range(self.Nq):
            term1 = math_func.lorentzian(self.omega_ij[:,:] - self.omega_alpha_q[q], self.Delta_alpha_q) 
            term2 = math_func.lorentzian(self.omega_ij[:,:] + self.omega_alpha_q[q], self.Delta_alpha_q) 
        
            self.G_1ph_value[:,:,q] = (1 / np.pi) * (term1 * self.n_alpha_q[q] + term2 * (self.n_alpha_q[q] + 1))

        return 
    

    def G_2ph(self):
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
        for q1 in range(self.Nq):
            for q2 in range(self.Nq):

                term1 = math_func.lorentzian(self.omega_ij[:,:] - self.omega_alpha_q[q1] - self.omega_alpha_q[q2], self.Delta_alpha_q) 
                term2 = math_func.lorentzian(self.omega_ij[:,:] + self.omega_alpha_q[q1] + self.omega_alpha_q[q2], self.Delta_alpha_q) 
                term3 = math_func.lorentzian(self.omega_ij[:,:] + self.omega_alpha_q[q1] - self.omega_alpha_q[q2], self.Delta_alpha_q) 
                term4 = math_func.lorentzian(self.omega_ij[:,:] - self.omega_alpha_q[q1] + self.omega_alpha_q[q2], self.Delta_alpha_q) 
    

                self.G_2ph_value[:,:,q1,q2] = (1 / np.pi) * (term1 * self.n_alpha_q[q1] * self.n_alpha_q[q2] + 
                                                             term2 * (self.n_alpha_q[q1] + 1) * (self.n_alpha_q[q2] + 1) +
                                                             term3 * self.n_alpha_q[q1] * (self.n_alpha_q[q2] + 1) +
                                                             term4 * (self.n_alpha_q[q1] + 1) * self.n_alpha_q[q2])


        return 



