[![status](https://joss.theoj.org/papers/6cd884ec29554707741cd8e700542a68/status.svg)](https://joss.theoj.org/papers/6cd884ec29554707741cd8e700542a68)


# SpinVibe


**SpinVibe** is an open-source Python package for simulating **spin–phonon coupling in molecular qubits** using first-principles calculations.  
It is desgin to study molecular qubits in solid-state systems.

Please take a look at the  [Documentation](https://victorchanglee.github.io/SpinVibe/)

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

## SpinVibe Installation Guide

### Step 1: Download Source Code

Download SpinVibe source code using the command:

```bash
git clone https://github.com/victorchanglee/SpinVibe.git
```

---

### Step 2: Navigate to Root Directory

```bash
cd SpinVibe
```

---

### Step 3-a: Install the Code from setup.py

Install the code with the command:

```bash
pip install -e .
```
alternatively

### Step 3-a: Install the Code from pyproject.toml


```bash
python -m build
pip install dist/spinvibe-1.0-py3-none-any.whl
```

Once the installation is succesfull, you can test it using the files provided in the test directory.
