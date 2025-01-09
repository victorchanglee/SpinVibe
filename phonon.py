import numpy as np
from constants import hbar,k_B

class phonon:
    def __init__(self,omega_ij, omega_alpha_q, Delta_alpha_q, T):
        
        self.hbar = hbar
        self.k_B = k_B

        self.omega_ij = omega_ij
        self.omega_alpha_q = omega_alpha_q
        self.Delta_alpha_q = Delta_alpha_q
        

        self.n_alpha_q = self.bose_einstein()

        self.G_1ph_value = 0.0
        self.G_2ph_value = 0.0

        self.G_1ph()
        self.G_2ph()

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

        
        if T == 0:
            return 0.0  # At T=0, the occupation number is zero
        else:
            return 1.0 / (np.exp(self.hbar * omega_alpha_q / (self.k_B * T)) - 1)


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
        term1 = self.Delta_alpha_q / (self.Delta_alpha_q**2 + (self.omega_ij - self.omega_alpha_q)**2)
        term2 = self.Delta_alpha_q / (self.Delta_alpha_q**2 + (self.omega_ij + self.omega_alpha_q)**2)
        
        self.G_1ph_value = (1 / np.pi) * (term1 * n_alpha_q + term2 * (n_alpha_q + 1))

        return 
    

     def G_2ph(self):
        """
        
        """


        return 



