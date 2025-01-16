import numpy as np
from scipy.optimize import curve_fit

def lorentzian(x, eta):

    return (eta / np.pi) / ( (x) ** 2 + eta ** 2)

def diagonalize(hamiltonian):
    """
    Diagonalize a Hamiltonian.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return eigenvalues, eigenvectors

def compute_derivative(x,fx):
    """
    Compute the derivative of f_x with reespect to x by polynomial fitting

    Returns: df_x / dx (array): Derivative of f_x with respect to x at 0.

    """
    
    # Step 2: Fit a polynomial to the data
    degree = 3  # Degree of the polynomial
    coefficients = np.polyfit(x, fx, degree)
    polynomial = np.poly1d(coefficients)

    # Step 3: Differentiate the polynomial
    derivative_polynomial = polynomial.deriv()

    # Step 4: Evaluate the derivative at specific points
    
    f_derivative = derivative_polynomial(0)



    return f_derivative

def compute_second_derivative(xy, fx):

    def poly_func(coords, a0, a1, a2, a3, a4, a5):
        x, y = coords
        return a0 + a1*x + a2*y + a3*x**2 + a4*y**2 + a5*x*y
    
    coords = (xy[:, 0], xy[:, 1])  # x and y displacements
    popt, _ = curve_fit(poly_func, coords, fx)

    a0, a1, a2, a3, a4, a5 = popt
    
    d2Hs_dx2 = 2 * a3
    d2Hs_dy2 = 2 * a4
    d2Hs_dxdy = a5

    

    return 

def energy_diff(eigenvalues):
    """
    Compute the energy difference.
    """

    energy_diff = eigenvalues[:, None] - eigenvalues[None, :]
    return energy_diff


