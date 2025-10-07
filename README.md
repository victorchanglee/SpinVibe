<style>
.container {
  display: flex;
  gap: 2rem;
  max-width: 1200px;
}

.sidebar {
  flex: 0 0 200px;
  position: sticky;
  top: 20px;
  height: fit-content;
}

.sidebar nav {
  background: #f5f5f5;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #ddd;
}

.sidebar h3 {
  margin-top: 0;
  font-size: 1rem;
  margin-bottom: 0.75rem;
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
  color: #333;
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
      <h3>Contents</h3>
      <ul>
        <li><a href="#step-1">1. Download Source</a></li>
        <li><a href="#step-2">2. Navigate Directory</a></li>
        <li><a href="#step-3">3. Install Code</a></li>
      </ul>
    </nav>
  </aside>

  <main class="content">

## SpinVibe Installation Guide

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
