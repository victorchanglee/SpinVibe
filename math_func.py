import numpy as np
from constants import k_B

def lorentzian(x, eta):

    return (eta / np.pi) / ( (x) ** 2 + eta ** 2)

def diagonalize(hamiltonian):
    """
    Diagonalize a Hamiltonian.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return eigenvalues, eigenvectors


def energy_diff(eigenvalues):
    """
    Compute the energy difference.
    """

    energy_diff = eigenvalues[:, None] - eigenvalues[None, :]
    return energy_diff


def mat(a, b, alpha,q,eigenvectors, V_alpha):

    tmp = np.dot(V_alpha[q,alpha,:, :],eigenvectors[b])
    tmp1 = np.dot(eigenvectors[a], tmp)

    return tmp1

def bose_einstein(omega_alpha_q, T):
        """
        Compute the Bose-Einstein occupation number.
        
        Parameters:
            omega_alpha_q (float): Phonon frequency (ω_{αq}).
            T (float): Temperature in Kelvin.
        
        Returns:
            float: Bose-Einstein occupation number (n̄_{αq}).
        """
        x = omega_alpha_q / (k_B * T)
        x = np.clip(x, None, 700)
        
        n_alpha_q = 1 / (np.exp(x) - 1)

        return n_alpha_q

def compute_derivative(x,fx,displacement=0.0,degree=5):
    """
    Compute the derivative of f_x with reespect to x by polynomial fitting

    Returns: df_x / dx (array): Derivative of f_x with respect to x at 0.

    """
    
    coefficients = np.polyfit(x, fx, degree)
    
    # Calculate derivative coefficients (using polyder)
    deriv_coeffs = np.polyder(coefficients)
    
    # Evaluate derivative at displacement
    dfdx = np.polyval(deriv_coeffs, displacement)

    return dfdx

def compute_trace(matrix):
    n = len(matrix)
    if n == 0 or len(matrix[0]) != n:
        raise ValueError("Matrix must be square.")
    return sum(matrix[i][i] for i in range(n))