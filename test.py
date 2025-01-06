import numpy as np

def zeeman_term_per_spin(B, g_tensors, spins, beta):
    """
    Compute the Zeeman interaction energy per spin.
    """
    zeeman_energies = []
    for i, spin in enumerate(spins):
        energy = beta[i] * np.dot(B, np.dot(g_tensors[i], spin))
        zeeman_energies.append(energy)
    return np.array(zeeman_energies)

def crystal_field_term_per_spin(B_lm, spin_operators, spins):
    """
    Compute the crystal field interaction energy per spin.
    """
    crystal_field_energies = []
    for i, spin in enumerate(spins):
        energy = 0.0
        S_magnitude = np.linalg.norm(spin)
        S_operators = spin_operators[i]
        for l in range(2, int(2 * S_magnitude) + 1):
            for m in range(-l, l + 1):
                energy += B_lm[i][l][m] * S_operators[(l, m)]
        crystal_field_energies.append(energy)
    return np.array(crystal_field_energies)

def exchange_term_per_spin(spins, J_tensors):
    """
    Compute the exchange interaction energy per spin.
    """
    N = len(spins)
    exchange_energies = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                exchange_energies[i] += np.dot(spins[i], np.dot(J_tensors[i, j], spins[j]))
    return exchange_energies

def compute_hamiltonian_per_spin(B, g_tensors, spins, beta, B_lm, spin_operators, J_tensors):
    """
    Compute the Hamiltonian contributions per spin.
    """
    # Zeeman interaction term per spin
    zeeman_energies = zeeman_term_per_spin(B, g_tensors, spins, beta)

    # Crystal field term per spin
    crystal_field_energies = crystal_field_term_per_spin(B_lm, spin_operators, spins)

    # Exchange interaction term per spin
    exchange_energies = exchange_term_per_spin(spins, J_tensors)

    # Total energy per spin
    total_energies = zeeman_energies + crystal_field_energies + 0.5 * exchange_energies
    return total_energies

# Example Inputs
B = np.array([0.1, 0.2, 0.3])  # Magnetic field vector
g_tensors = [np.eye(3) for _ in range(3)]  # Identity g-tensors for 3 spins
spins = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]  # Spin vectors
beta = [1.0, 1.5, 2.0]  # Coupling constants

# Crystal field coefficients and operators
B_lm = [{l: {m: 0.1 for m in range(-l, l + 1)} for l in range(2, 4)} for _ in range(3)]
spin_operators = [{(l, m): 0.1 for l in range(2, 4) for m in range(-l, l + 1)} for _ in range(3)]

# Exchange coupling tensors
J_tensors = np.zeros((3, 3, 3, 3))  # Example 3x3 spin system
J_tensors[0, 1] = np.eye(3)  # Interaction between spin 0 and 1
J_tensors[1, 2] = np.eye(3)  # Interaction between spin 1 and 2

# Compute the Hamiltonian energy per spin
H_s_per_spin = compute_hamiltonian_per_spin(B, g_tensors, spins, beta, B_lm, spin_operators, J_tensors)

print(f"Hamiltonian energy per spin: {H_s_per_spin}")