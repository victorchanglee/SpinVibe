<style>
.container {
  display: flex;
  gap: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.sidebar {
  flex: 0 0 220px;
  position: sticky;
  top: 20px;
  height: fit-content;
}

.sidebar nav {
  background: #f5f5f5;
  padding: 1.25rem;
  border-radius: 8px;
  border: 1px solid #ddd;
}

.sidebar h3 {
  margin-top: 0;
  font-size: 1rem;
  margin-bottom: 1rem;
  color: #333;
}

.sidebar ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.sidebar li {
  margin: 0.5rem 0;
}

.sidebar a {
  text-decoration: none;
  color: #555;
  font-size: 0.9rem;
}

.sidebar a:hover {
  color: #0066cc;
}

.content {
  flex: 1;
  min-width: 0;
}
</style>

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
