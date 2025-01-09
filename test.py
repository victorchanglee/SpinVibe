import numpy as np

# Constants
hbar = 1.0545718e-34  # Planck's constant [J·s]

# Input data
Ns = 4  # Number of eigenstates
Nq = 3  # Number of phonon modes
omega_q = np.array([1.0, 1.5, 2.0])  # Phonon frequencies (example) [eV]
omega_ab = np.random.rand(Ns, Ns)  # Energy differences [eV]
mass_term = np.array([1.0, 1.2, 1.5])  # Example mass weights

# Interaction matrix elements V^{alpha q}_{aj} (random example)
V_alpha_q = np.random.rand(Ns, Ns, Nq)

# Example Green's function G^{1-ph} (frequency-dependent)
def G_1_ph(omega_1, omega_2):
    return 1 / (omega_1 - omega_2 + 1j * 1e-3)  # Example Green's function

# Redfield tensor initialization
R_tensor = np.zeros((Ns, Ns, Ns, Ns), dtype=np.complex128)

# Calculate the Redfield tensor
for a in range(Ns):
    for b in range(Ns):
        for c in range(Ns):
            for d in range(Ns):
                # Sum over alpha, q
                for alpha in range(3):  # Assuming 3 spatial directions
                    for q in range(Nq):
                        # First summation term
                        term_1 = 0
                        for j in range(Ns):
                            term_1 += (V_alpha_q[a, j, q] * V_alpha_q[j, c, q] *
                                       G_1_ph(omega_ab[j, c], omega_q[q]))
                        
                        # Second summation term
                        term_2 = V_alpha_q[a, c, q] * V_alpha_q[d, b, q] * G_1_ph(omega_ab[b, d], omega_q[q])
                        
                        # Third summation term
                        term_3 = 0
                        for j in range(Ns):
                            term_3 += (V_alpha_q[c, d, q] * V_alpha_q[a, j, q] *
                                       G_1_ph(omega_ab[j, d], omega_q[q]))
                        
                        # Combine terms and update the tensor
                        R_tensor[a, b, c, d] += (-np.pi / (2 * hbar) *
                            (term_1 - term_2 + term_3) * np.sqrt(1 / (Ns * omega_q[q] * mass_term[q % len(mass_term)])))

# Output the result
print("Redfield tensor R_ab,cd:")
print(R_tensor)