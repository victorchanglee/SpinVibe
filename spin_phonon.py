import numpy as np


""""
Calculation run file

"""

class spin_phonon:
    def __init__(self,spins,B, g_tensors, beta,D_tensor):
        
        
        self.spins = spins
        self.N = len(self.spins)  # Number of spins

        self.B = B # external magnetic field
        self.g_tensors = g_tensors
        self.beta = beta
        self.D_tensor = D_tensor

        self.H_s = np.zeros([self.N],dtype=np.float64)

        self.init_spin_hamiltonian()

    
        return
    
    def read_file():
        return
    
    def init_spin_hamiltonian(self):
        """
        Compute the Hamiltonian H_s for the spin system.

        Parameters:
            B (array): External magnetic field vector (3D).
            g_tensors (list): List of 3x3 g-tensors for each spin.
            spins (list): List of spin vectors (3D) for each site.
            beta (list): Coupling constants for each spin.
            D_tensor (array): NxN array of 3x3 D-tensors for spin-spin interactions.

        Returns:
            float: Value of the Hamiltonian.
        """
        
        

        # First term: Interaction with external magnetic field
        for i in range(self.N):
            self.H_s[i] = self.beta[i] * np.dot(self.B, np.dot(self.g_tensors[i], self.spins[i]))
            
        # Second term: Spin-spin interactions
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.H_s[i] += 0.5 * np.dot(self.spins[i], np.dot(self.D_tensor[i, j], self.spins[j]))

        return 
    