import numpy as np

def compute_hamiltonian(B, g_tensors, spins, beta, D_tensor):
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
    N = len(spins)  # Number of spins
    H_s = 0.0

    # First term: Interaction with external magnetic field
    for i in range(N):
        H_s += beta[i] * np.dot(B, np.dot(g_tensors[i], spins[i]))

    # Second term: Spin-spin interactions
    for i in range(N):
        for j in range(N):
            if i != j:
                H_s += 0.5 * np.dot(spins[i], np.dot(D_tensor[i, j], spins[j]))

    return H_s

# Example inputs
B = np.array([0.1, 0.2, 0.3])  # Magnetic field vector
g_tensors = [np.eye(3) for _ in range(3)]  # Identity g-tensors for 3 spins
spins = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]  # Spin vectors
beta = [1.0, 1.5, 2.0]  # Coupling constants
D_tensor = np.zeros((3, 3, 3, 3))  # Spin-spin interaction tensor
D_tensor[0, 1] = np.eye(3)  # Example interaction between spin 0 and 1
D_tensor[1, 0] = np.eye(3)  # Symmetric interaction

# Compute the Hamiltonian
H_s = compute_hamiltonian(B, g_tensors, spins, beta, D_tensor)

print(f"Hamiltonian H_s: {H_s}")