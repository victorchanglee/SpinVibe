import numpy as np

def delta_lorentzian(x, xc, eta):

    return (eta / np.pi) / ( (x-xc) ** 2 + eta ** 2)

def diagonalize(hamiltonian):
    """
    Diagonalize a Hamiltonian.
    """

    # Diagonalize the Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return eigenvalues, eigenvectors