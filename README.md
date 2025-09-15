# SpinVibe

[![Draft PDF](https://github.com/yourusername/SpinVibe/actions/workflows/draft-pdf.yml/badge.svg)](https://github.com/yourusername/SpinVibe/actions/workflows/draft-pdf.yml)

**SpinVibe** is an open-source Python package for simulating **spin–phonon coupling in molecular qubits** using first-principles calculations.  
It is designed to support materials discovery for quantum information and quantum sensing applications.

---

## Features
- Parse and analyze molecular structures from **DFT outputs**.
- Compute **spin–phonon coupling** parameters.
- Tools for **vibronic Hamiltonians** and **relaxation rate calculations**.
- Interfaces with standard electronic-structure codes (e.g., Quantum ESPRESSO, CASTEP, BerkeleyGW).
- JOSS-compatible paper build (see [docs/paper.md](docs/paper.md)).

---

## Requirements
The code was written using the following Python Libraries:

- python                    3.9.21
- numpy                     1.26.4
- scipy                     1.13.1
- mpi4py                    4.0.2
- h5py                      3.12.1
- tqdm                      4.67.1 (optional for tracking parallerization performance)
