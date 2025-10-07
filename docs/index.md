

<div class="container">
  <aside class="sidebar">
    <nav>
      <h3>Table of Contents</h3>
      <ul>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#inputs">Inputs</a></li>
        <li><a href="#output">Output</a></li>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#installation">Installation Guide</a></li>
        <li style="margin-left: 1rem;"><a href="#step-1">Step 1: Download</a></li>
        <li style="margin-left: 1rem;"><a href="#step-2">Step 2: Navigate</a></li>
        <li style="margin-left: 1rem;"><a href="#step-3">Step 3: Install</a></li>
      </ul>
    </nav>
  </aside>

  <main class="content">

# <span id="overview">SpinVibe</span>

**SpinVibe** is an open-source Python package for simulating **spin–phonon coupling in molecular qubits** using first-principles calculations.  
It is designed to study molecular qubits in solid-state systems.

---

## <span id="inputs">Inputs</span>

- Solid-state phonons and eigenvectors
- Molecular spin-hamiltonian parameters (e.g. g-factor, zero-field splitting tensor)
- Parameters: Temperature, Polarization, external magnetic field, etc.

## <span id="output">Output</span>

- Time evolution of the spin density and magnetization
- Spin-phonon relaxation time (T1)

---

## <span id="requirements">Requirements</span>

The code was written using the following Python Libraries:
- python                    3.9.21
- numpy                     1.26.4
- scipy                     1.13.1
- mpi4py                    4.0.2
- h5py                      3.12.1

## <span id="installation">SpinVibe Installation Guide</span>

### <span id="step-1">Step 1: Download Source Code</span>

Download SpinVibe source code using the command:

```bash
git clone https://github.com/victorchanglee/SpinVibe.git
```

---

### <span id="step-2">Step 2: Navigate to Root Directory</span>

```bash
cd SpinVibe
```

---

### <span id="step-3">Step 3: Install the Code</span>

Install the code with the command:

```bash
pip install -e .
```

  </main>
</div>
