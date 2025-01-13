import numpy as np
from constants import hbar

class coupling:
    def __init__(self,Hs_xi, N_q, q_vector, omega_alpha_q, masses, R_vectors, L_vectors, disp):
        
        self.Hs_xi = Hs_xi

        self.Ns = len(Hs_xi[0,0,0,:])
        

        self.N_q = N_q
        self.q_vector = q_vector
        self.omega_alpha_q = omega_alpha_q
        self.masses = masses
        self.R_vectors = R_vectors
        self.L_vectors = L_vectors

        self.N_cells = len(R_vectors)
        self.N_atoms = len(L_vectors)
        self.hbar = hbar
        self.disp = disp

        self.dh_dq = np.zeros((self.Ns,3), dtype=np.complex128)

        self.compute_coupling()

        return 
    
    def compute_coupling(self):
        """
        Compute the spin-phonon coupling energy.

        Parameters:
            Q_alpha_q (tuple): Phonon mode index and wave vector q.
            omega_alpha_q (float): Phonon frequency for mode alpha at q.
            masses (array): Masses of the atoms.
            R_vectors (array): Position vectors of the cells.
            L_vectors (array): Displacement vectors of the atoms.
            dHs_dx (array): Derivatives of the spin Hamiltonian with respect to atomic displacements.

        Returns:
            float: Spin-phonon coupling energy.
        """

        mass_term = np.sqrt(self.hbar / (self.N_q * self.omega_alpha_q * self.masses))  # sqrt(hbar / (N_q * omega * m))
        phase_factor = np.exp(1j * np.einsum('i,ji->ji', self.q_vector, self.R_vectors))  # e^{i q . R_l}

        # Compute the first derivative of the Hamiltonian with respect to atomic displacements  
        
        dh_dx = np.zeros((self.N_atoms, self.Ns, 3), dtype=np.complex128)
       
        for n in range(self.N_atoms):
            for i in range(3):
                for s in range(self.Ns):    
                    x = self.disp
                    f_x = self.Hs_xi[n,i,:,s]

                    dh_dx[n,s,i] = self.compute_dHs_dx(x,f_x)

        
        tmp1 = np.einsum('ij,ikj->ikj', self.L_vectors, dh_dx) 
        tmp2 = np.einsum('i,ikj->kj', mass_term, tmp1) #Sum over atoms
        tmp3 = np.einsum('lj,kj->kj', phase_factor, tmp2) #Sum over cells

        self.dh_dq = tmp3 

        return
    
    
    def compute_dHs_dx(self,x,f_x):
        """
        Compute the derivative of f_x with reespect to x by polynomial fitting

        Returns: df_x / dx (array): Derivative of f_x with respect to x at 0.

        """
        
        # Step 2: Fit a polynomial to the data
        degree = 3  # Degree of the polynomial
        coefficients = np.polyfit(x, f_x, degree)
        polynomial = np.poly1d(coefficients)

        # Step 3: Differentiate the polynomial
        derivative_polynomial = polynomial.deriv()

        # Step 4: Evaluate the derivative at specific points
        
        f_derivative = derivative_polynomial(0)



        return f_derivative




