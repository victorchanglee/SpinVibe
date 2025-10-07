# SpinVibe


**SpinVibe** is an open-source Python package for simulating **spin–phonon coupling in molecular qubits** using first-principles calculations.  
It is desgin to study molecular qubits in solid-state systems.

---

## Inputs
- Solid-state phonons and eigenvectors
- Molecular spin-hamiltonian parameters (e.g. g-factor, zero-field splitting tensor)
- Parameters: Temperature, Polarization, external magnetic field, etc.

## Output
- Time evolution of the spin density and magnetization
- Spin-phonon relaxation time (T1)

---

## Requirements
The code was written using the following Python Libraries:

- python                    3.9.21
- numpy                     1.26.4
- scipy                     1.13.1
- mpi4py                    4.0.2
- h5py                      3.12.1
- tqdm                      4.67.1 (optional for tracking parallerization performance)
