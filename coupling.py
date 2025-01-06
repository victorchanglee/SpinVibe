import numpy as np

def first_finite_diff(f, x, delta):
    """
    Compute the first derivative of a function f using the finite difference method.

    Parameters:
        f (function): Function to differentiate.
        x (float or array): Point(s) at which to evaluate the first derivative.
        delta (float): Small step size for finite difference.

    Returns:
        array: First derivative of f at x.
    """
    derivative_first = (f(x + delta) - f(x - delta)) / (delta ** 2)

    return derivative_first

def second_finite_diff(f,x,delta):
    """
    Compute the second derivative of a function f using the finite difference method.

    Parameters:
        f (function): Function to differentiate.
        x (float or array): Point(s) at which to evaluate the second derivative.
        delta (float): Small step size for finite difference.

    Returns:
        float or array: Second derivative of f at x.
    """
    derivative_second = (f(x + delta) - 2 * f(x) + f(x - delta)) / (delta ** 2)

    return derivative_second

def compute_prefactor(N_q, omega_alpha_q, mass, q_vector, R_vectors, mode_amplitude):
    """
    Compute the prefactor for the given parameters.
    Ry atomic units, hbar = 1 
    Parameters:
        
        N_q (int): Number of q-points.
        omega_alpha_q (float): Phonon frequency for mode alpha at q.
        mass: Mass of the atom.
        q_vector (array): Wave vector q (3D).
        R_vectors (list or array): Position vectors of the cells.
        mode_amplitude (list or array): Mode amplitudes L_alpha_i^q for each atom.

    Returns:
        list: Prefactor for each cell and atom.
    """
    prefactors = []
    for l, R_l in enumerate(R_vectors):

        phase_factor = np.exp(1j * np.dot(q_vector, R_l))  # e^{i q . R_l}
        mass_term = np.sqrt(1 / (N_q * omega_alpha_q * mass))  # sqrt(hbar / (N_q * omega * m))
        prefactor = mass_term * phase_factor * mode_amplitude

        prefactors.append(prefactor)
    return np.array(prefactors)



