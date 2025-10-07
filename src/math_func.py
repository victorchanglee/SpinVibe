import numpy as np
from constants import k_B, kB_SI
from numpy.polynomial.polynomial import polyfit, polyval2d, Polynomial

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

def finite_difference(f_plus, f_minus, delta):
    """
    Computes the numerical gradient at 0 using central difference.

    Parameters:
        f_plus  : float or np.array
            Function value at +delta
        f_zero  : float or np.array
            Function value at 0
        f_minus : float or np.array
            Function value at -delta
        delta   : float
            Displacement step size

    Returns:
        float or np.array: Estimated derivative at 0
    """
    return (f_plus - f_minus) / (2 * delta)

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


def compute_trace(matrix):
    n = len(matrix)
    if n == 0 or len(matrix[0]) != n:
        raise ValueError("Matrix must be square.")
    return sum(matrix[i][i] for i in range(n))

def tensor_traceless(tensor):
    """
    Transforms a 3x3 numpy array into a traceless symmetric tensor.

    Args:
        tensor (numpy.ndarray): A 3x3 numpy array (assumed to be symmetric).

    Returns:
        numpy.ndarray: A 3x3 traceless symmetric numpy array.
                       Returns None if the input is not a 3x3 array.
    """
    if tensor is None or tensor.shape != (3, 3):
        print("Error: Input tensor must be a 3x3 NumPy array.")
        return None

    # Ensure it's symmetric (important if the raw data isn't perfectly symmetric due to parsing or small numerical errors)
    # This step is good practice, but for ORCA output, it's usually already symmetric
    tensor = (tensor + tensor.T) / 2

    trace = np.trace(tensor)
    avg_trace = trace / 3.0

    traceless_tensor = tensor - avg_trace * np.identity(3)

    return traceless_tensor

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

