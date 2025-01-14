import numpy as np


""""
Calculation run file

"""

class hamiltonian:
    def __init__(self, B, S, dim, g_tensors, beta, J_tensors):
        
        self.B = B
        self.S = S
        self.dim = dim
        self.N = int(2*self.S + 1)
        self.g_tensors = g_tensors
        self.beta = beta
        self.J_tensors = J_tensors

        self.m = np.arange(-self.S, self.S+1, 1)  # m values
        self.Sx = np.zeros((self.N, self.N),dtype=np.complex128)
        self.Sy = np.zeros((self.N, self.N),dtype=np.complex128)
        self.Sz = np.zeros((self.N, self.N),dtype=np.complex128)

        self.spin_operators()
    # Total dimension of the Hilbert space
        self.hdim = self.N ** self.dim

        

        self.Hs = np.zeros((self.hdim, self.hdim), dtype=np.complex128)  # Initialize spin Hamiltonian terms with zeros
        self.zeeman_energy = np.zeros((self.hdim, self.hdim), dtype=np.complex128)
        self.exchange_energy = np.zeros((self.hdim, self.hdim), dtype=np.complex128)
        
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
        self.zeeman()


        # Exchange interaction term
        self.exchange_interaction()

        # Field splitting term
        #self.field_energy = np.zeros((self.h_dim, self.h_dim), dtype=np.complex128)
        #self.field_splitting()

        # Total Hamiltonian energy
        self.Hs = self.zeeman_energy + 0.5 * self.exchange_energy #+ self.field_energy 
    
        return
    
    def spin_operators(self):
        """
        Returns the spin operators Sx, Sy, Sz for a given spin quantum number S.

        Spin operators from: https://easyspin.org/documentation/spinoperators.html
        
        """
        
        
        for i, l in enumerate(self.m):  # m is the row index
            for j, l_prime in enumerate(self.m):  # m' is the column index
                # Sx
                if l_prime == l + 1 or l_prime == l - 1:
                    self.Sx[j, i] = 0.5 * np.sqrt(self.S * (self.S + 1) - l * l_prime)

                # Sy
                if l_prime == l + 1:
                    self.Sy[j, i] = -0.5j * np.sqrt(self.S * (self.S + 1) - l * l_prime)
                elif l_prime == l - 1:
                    self.Sy[j, i] = 0.5j * np.sqrt(self.S * (self.S + 1) - l * l_prime)

                # Sz
                if l_prime == l:
                    self.Sz[j, i] = l

        return 


    def zeeman(self):
        """
        Compute the Zeeman interaction energy.
        """


        for i in range(self.N):
            Sx_total = np.eye(1)
            Sy_total = np.eye(1)
            Sz_total = np.eye(1)
            
            for j in range(self.dim):
                if j == i:
                    Sx_total = np.kron(Sx_total, self.Sx)
                    Sy_total = np.kron(Sy_total, self.Sy)
                    Sz_total = np.kron(Sz_total, self.Sz)
                else:
                    Sx_total = np.kron(Sx_total, np.eye(self.N))
                    Sy_total = np.kron(Sy_total, np.eye(self.N))
                    Sz_total = np.kron(Sz_total, np.eye(self.N))
            

            # Add g-tensor contributions
            self.zeeman_energy += self.beta[i] * self.B[0] * self.g_tensors[i,i][0] * Sx_total
            self.zeeman_energy += self.beta[i] * self.B[1] * self.g_tensors[i,i][1] * Sy_total
            self.zeeman_energy += self.beta[i] * self.B[2] * self.g_tensors[i,i][2] * Sz_total

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
                    self.field_energy += self.B_lm[i][l][m] * S_operators[(l, m)]
        return 

    def exchange_interaction(self):
        """
        Compute the exchange interaction energy.
        """

        exchange_energy = np.zeros((self.hdim, self.hdim), dtype=np.complex128)
        
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    J_matrix = self.J_tensors[i][j]
                    for a, op1 in enumerate([self.Sx,self.Sy, self.Sz]):  # Index over spin components
                        for b, op2 in enumerate([self.Sx, self.Sy, self.Sz]):
                            J_comp = J_matrix[a, b]
                            # Coupling strength between components
                            Si = np.eye(1)
                            Sj = np.eye(1)
                            for k in range(self.dim):
                                if k == i:
                                    Si = np.kron(Si, op1)  # Embed spin operator op1 for spin i
                                else:
                                    Si = np.kron(Si, np.eye(self.N))
                                if k == j:
                                    Sj = np.kron(Sj, op2)  # Embed spin operator op2 for spin j
                                else:
                                    Sj = np.kron(Sj, np.eye(self.N))

                            self.exchange_energy += J_comp * np.dot(Si,Sj)  # Add the interaction term
                
        
        return 

    

    
    