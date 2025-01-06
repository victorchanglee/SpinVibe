import numpy as np


""""
Calculation run file

"""

class s_hamiltonian:
    def __init__(self, B, g_tensors, spins, beta, B_lm,spin_operators, J_tensors):
        
        self.B = B
        self.g_tensors = g_tensors
        self.spins = spins
        self.beta = beta
        self.B_lm = B_lm
        self.spin_operators = spin_operators
        self.J_tensors = J_tensors
        self.N = len(spins)  # Number of spins
        self.H_s = np.zeros([self.N], dtype=np.float64)  # Initialize spin Hamiltonian terms with zeros
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

        # Zeeman interaction term
        self.zeeman_energy = np.zeros(self.N, dtype=np.float64)
        self.zeeman()

        # Field splitting term
        self.field_energy = np.zeros(self.N, dtype=np.float64)
        self.field_splitting()

        # Exchange interaction term
        self.exchange_energy= np.zeros(self.N, dtype=np.float64)
        self.exchange_interaction()

        # Total Hamiltonian energy
        self.H_s = self.zeeman_energy + self.field_energy + 0.5 * self.exchange_energy
    
        return

    def zeeman(self):
        """
        Compute the Zeeman interaction energy.
        """

        for i in range(self.N):
            self.zeeman_energy[i] += self.beta[i] * np.dot(self.B, np.dot(self.g_tensors[i], self.spins[i]))

        return

    def field_splitting(self):
        """
        Compute the crystal field interaction energy.
        """

        for i in range(self.N):
            S_i = self.spins[i]
            S_magnitude = np.linalg.norm(S_i)
            S_operators = self.spin_operators[i]
            for l in range(2, int(2 * S_magnitude) + 1):
                for m in range(-l, l + 1):
                    self.field_energy[i] += self.B_lm[i][l][m] * S_operators[(l, m)]
        return 

    def exchange_interaction(self):
        """
        Compute the exchange interaction energy.
        """

        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.exchange_energy += np.dot(self.spins[i], np.dot(self.J_tensors[i, j], self.spins[j]))
        return 

    

    
    