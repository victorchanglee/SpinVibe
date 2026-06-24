import numpy as np
from .constants import k_B, kB_SI, hbar_SI
from numpy.polynomial.polynomial import polyfit, polyval2d, Polynomial

def broadening(x, eta):

    return (eta / np.pi) / ( (x) ** 2 + eta ** 2)
    #return (1/eta * (np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x/eta)**2)

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

    if omega_alpha_q <= 0:
        n_alpha_q = 0.0
    else:
        x = (hbar_SI * omega_alpha_q) / (kB_SI * T)
        x = np.clip(x, None, 700)  # Prevent overflow
        n_alpha_q = 1.0 / (np.exp(x) - 1.0)


    return n_alpha_q


def compute_derivative(x,fx,displacement=0.0,degree=3):
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


def compute_second_derivative(x, fx, poly_degree=3):
    """
    Fit a 2D polynomial to the function values and compute the mixed second derivative at (0, 0).
    
    Parameters:
        x: np.ndarray of shape (2, N) — x[0] contains x values, x[1] contains y values.
        fx: np.ndarray of shape (M, M) — function values on the x-y grid.
        poly_degree: int — degree of the 2D polynomial.

    Returns:
        float: second derivative ∂²f/∂x∂y at (0, 0).
    """
    x_vals, y_vals = x[0], x[1]
    
    # Handle potential dimension mismatch by using the actual fx dimensions
    if fx.shape[0] != len(x_vals) or fx.shape[1] != len(y_vals):
        # Trim or extend the coordinate arrays to match fx dimensions
        n_x, n_y = fx.shape
        if len(x_vals) > n_x:
            x_vals = x_vals[:n_x]
        if len(y_vals) > n_y:
            y_vals = y_vals[:n_y]
        
        # If fx is larger, we need to handle this case
        if len(x_vals) < n_x or len(y_vals) < n_y:
            raise ValueError(f"Function values shape {fx.shape} is larger than coordinate arrays ({len(x_vals)}, {len(y_vals)})")
    
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    
    # Flatten all arrays consistently
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = fx.ravel()

    # Construct polynomial basis
    A = []
    for i in range(poly_degree + 1):
        for j in range(poly_degree + 1 - i):
            A.append(X_flat**i * Y_flat**j)
    A = np.column_stack(A)


    coeffs, *_ = np.linalg.lstsq(A, Z_flat, rcond=None)
    
    deriv = 0.0
    idx = 0
    for i in range(poly_degree + 1):
        for j in range(poly_degree + 1 - i):
            if i == 1 and j == 1:  # Only the x*y term contributes at (0,0)
                deriv += coeffs[idx]
            idx += 1

    return deriv



def rotate_polarization(axis, theta):
    """
    Create 3D rotation matrix using Rodrigues' formula
    axis: unit vector (nx, ny, nz)
    theta: rotation angle in radians
    """
    axis = axis / np.linalg.norm(axis)  # normalize
    nx, ny, nz = axis
    c = np.cos(theta)
    s = np.sin(theta)
    
    R = np.array([
        [c + nx**2*(1-c),     nx*ny*(1-c) - nz*s,  nx*nz*(1-c) + ny*s],
        [ny*nx*(1-c) + nz*s,  c + ny**2*(1-c),     ny*nz*(1-c) - nx*s],
        [nz*nx*(1-c) - ny*s,  nz*ny*(1-c) + nx*s,  c + nz**2*(1-c)]
    ])
    return R

