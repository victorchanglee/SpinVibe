import numpy as np

def delta_lorentzian(x, xc, eta):

    return (eta / np.pi) / ( (x-xc) ** 2 + eta ** 2)

def diagonalize(hamiltonian):
    """
    Diagonalize a Hamiltonian.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return eigenvalues, eigenvectors

def compute_derivative(x,f_x):
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


def energy_diff(eigenvalues):
    """
    Compute the energy difference.
    """

    energy_diff = eigenvalues[:, None] - eigenvalues[None, :]
    return energy_diff