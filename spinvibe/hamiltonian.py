import numpy as np


""""
Calculation run file

"""

class hamiltonian:
    def __init__(self, B, S, g_tensors, D_tensors):
        
        self.B = B
        self.S = S
        self.Ns = int(2*self.S + 1)
        self.g_tensors = g_tensors

        self.D_tensors = D_tensors

        self.m = np.arange(-self.S, self.S+1, 1)  # m values
        self.Sx = np.zeros((self.Ns, self.Ns),dtype=np.complex128)
        self.Sy = np.zeros((self.Ns, self.Ns),dtype=np.complex128)
        self.Sz = np.zeros((self.Ns, self.Ns),dtype=np.complex128)

        self.spin_operators()
    # Total dimension of the Hilbert space
        self.hdim = self.Ns 
                

        self.Hs = np.zeros((self.hdim, self.hdim), dtype=np.complex128)  # Initialize spin Hamiltonian terms with zeros
        self.zeeman_energy = np.zeros((self.hdim, self.hdim), dtype=np.complex128)
        self.zfs_energy = np.zeros((self.hdim, self.hdim), dtype=np.complex128)
        
        # Zeeman interaction term
        self.zeeman()

        # ZFS interaction term
        self.zfs_interaction()

        self.Hs = self.zeeman_energy + self.zfs_energy 
        
        return
    

    
    def spin_operators(self):
        """
        Construct spin operators using ladder operator formalism.
        
        Standard formulas:
        S+ = Sx + iSy  (raising operator)
        S- = Sx - iSy  (lowering operator)
        
        S+ |S, m⟩ = ℏ√[S(S+1) - m(m+1)] |S, m+1⟩
        S- |S, m⟩ = ℏ√[S(S+1) - m(m-1)] |S, m-1⟩
        """
        
        for i, m in enumerate(self.m):  # i is row index (bra)
            for j, m_prime in enumerate(self.m):  # j is column index (ket)
                
                # Sz operator (diagonal)
                if i == j:
                    self.Sz[i, j] = m
                
                # S+ operator: connects |m⟩ to |m+1⟩
                if m_prime == m + 1:
                    self.Sx[i, j] = 0.5 * np.sqrt(self.S * (self.S + 1) - m * (m + 1))
                    self.Sy[i, j] = -0.5j * np.sqrt(self.S * (self.S + 1) - m * (m + 1))
                
                # S- operator: connects |m⟩ to |m-1⟩
                elif m_prime == m - 1:
                    self.Sx[i, j] = 0.5 * np.sqrt(self.S * (self.S + 1) - m * (m - 1))
                    self.Sy[i, j] = 0.5j * np.sqrt(self.S * (self.S + 1) - m * (m - 1))
    
        return

    def zeeman(self):
        """
        Compute the Zeeman interaction energy.
        """
        
        zeeman_x = self.B[0] * self.g_tensors[0,0] * self.Sx
        zeeman_y = self.B[1] * self.g_tensors[1,1] * self.Sy
        zeeman_z = self.B[2] * self.g_tensors[2,2] * self.Sz

        
        self.zeeman_energy = zeeman_x + zeeman_y + zeeman_z



    def zfs_interaction(self):
        """
        Compute the zero field splitting interaction energy.
        """

        spin_ops = {'x': self.Sx, 'y': self.Sy, 'z': self.Sz}
        components = ['x', 'y', 'z']
            
        for i, a in enumerate(components):
            for j, b in enumerate(components):
                Sa = spin_ops[a]
                Sb = spin_ops[b]
                self.zfs_energy +=  self.D_tensors[i,j] * np.dot(Sa, Sb)

        
        return 

    

    
    
