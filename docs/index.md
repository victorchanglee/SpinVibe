
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpinVibe Documentation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #404040;
        }


        /* Main Container */
        .container {
            display: flex;
            margin-top: 50px;
            min-height: calc(100vh - 50px);
        }

        /* Sidebar */
        .sidebar {
            width: 300px;
            background: #f9f9f9;
            border-right: 1px solid #e1e4e5;
            position: fixed;
            left: 0;
            top: 50px;
            bottom: 0;
            overflow-y: auto;
            padding: 20px 0;
        }

        .sidebar-content {
            padding: 0 20px;
        }

        .sidebar h3 {
            color: #2980b9;
            font-size: 16px;
            margin: 20px 0 10px 0;
            font-weight: 600;
        }

        .sidebar ul {
            list-style: none;
            margin: 0 0 15px 0;
        }

        .sidebar li {
            margin: 5px 0;
        }

        .sidebar a {
            color: #404040;
            text-decoration: none;
            display: block;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 14px;
            transition: background 0.2s;
        }

        .sidebar a:hover {
            background: #e1e4e5;
        }

        .sidebar .sub-item {
            margin-left: 15px;
            font-size: 13px;
        }

        /* Main Content */
        .main-content {
            margin-left: 300px;
            padding: 40px 60px;
            max-width: 1000px;
            flex: 1;
        }

        .main-content h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 20px;
            font-weight: 600;
        }

        .main-content h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin: 40px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #e1e4e5;
            font-weight: 600;
        }

        .main-content h3 {
            color: #2c3e50;
            font-size: 1.3em;
            margin: 30px 0 15px 0;
            font-weight: 600;
        }

        .main-content p {
            margin-bottom: 15px;
            line-height: 1.7;
        }

        .main-content ul {
            margin: 15px 0 15px 30px;
        }

        .main-content li {
            margin: 8px 0;
        }

        .main-content code {
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #e74c3c;
        }

        .main-content pre {
            background: #f8f8f8;
            border: 1px solid #e1e4e5;
            border-left: 3px solid #2980b9;
            padding: 15px;
            border-radius: 3px;
            overflow-x: auto;
            margin: 20px 0;
        }

        .main-content pre code {
            background: none;
            padding: 0;
            color: #333;
        }

        .note {
            background: #e7f2fa;
            border-left: 4px solid #2980b9;
            padding: 15px;
            margin: 20px 0;
            border-radius: 3px;
        }

        .note-title {
            font-weight: 600;
            color: #2980b9;
            margin-bottom: 8px;
        }

        hr {
            border: none;
            border-top: 1px solid #e1e4e5;
            margin: 30px 0;
        }

        /* Footer Navigation */
        .footer-nav {
            display: flex;
            justify-content: space-between;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e1e4e5;
        }

        .footer-nav a {
            color: #2980b9;
            text-decoration: none;
            padding: 10px 15px;
            border: 1px solid #2980b9;
            border-radius: 3px;
            transition: all 0.2s;
        }

        .footer-nav a:hover {
            background: #2980b9;
            color: white;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }
            .main-content {
                margin-left: 0;
                padding: 20px;
            }
        }
    </style>
</head>
<body>
   

    <!-- Main Container -->
    <div class="container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-content">
                <h3>Getting Started</h3>
                    <li><a href="#overview">Overview</a></li>
                    <li><a href="#installation">Installation</a></li>

                <h3>User Guide</h3>

                    <li><a href="#inputs">Inputs</a></li>
                    <li><a href="#output">Output</a></li>
                    <li><a href="#requirements">Requirements</a></li>
                    <li><a href="#example">Example</a></li>

                <h3>Additional Resources</h3>
                    <li><a href="#contributing">Contributing</a></li>
                    <li><a href="#license">License</a></li>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="main-content">
            <h1 id="overview">SpinVibe</h1>
            
            <p><strong>SpinVibe</strong> is an open-source Python package for simulating <strong>spin–phonon coupling in molecular qubits</strong> using first-principles calculations. It is designed to study molecular qubits in solid-state systems.</p>

            <div class="note">
                <div class="note-title">Note</div>
                <p>SpinVibe requires Python 3.9 or higher and several scientific computing libraries. See the <a href="#requirements">Requirements</a> section for details.</p>
            </div>

            <hr>

            <h2 id="inputs">Inputs</h2>
            <p>SpinVibe accepts the following inputs for simulations:</p>
            <ul>
                <li>Solid-state phonons and eigenvectors</li>
                <li>Molecular spin-hamiltonian parameters (e.g. g-factor, zero-field splitting tensor)</li>
                <li>Parameters: Temperature, Polarization, external magnetic field, etc.</li>
            </ul>

            <h2 id="output">Output</h2>
            <p>The package provides the following outputs:</p>
            <ul>
                <li>Time evolution of the spin density and magnetization</li>
                <li>Spin-phonon relaxation time (T1)</li>
            </ul>

            <hr>

            <h2 id="requirements">Requirements</h2>
            <p>The code was written using the following Python Libraries:</p>
            <ul>
                <li>python 3.9.21</li>
                <li>numpy 1.26.4</li>
                <li>scipy 1.13.1</li>
                <li>mpi4py 4.0.2</li>
                <li>h5py 3.12.1</li>
            </ul>

            <h2 id="installation">Installation Guide</h2>

               Step 1: Download Source Code
            <p>Download SpinVibe source code using the command:</p>
            <pre><code>git clone https://github.com/victorchanglee/SpinVibe.git</code></pre>

            Step 2: Navigate to Root Directory
            <pre><code>cd SpinVibe</code></pre>

            Step 3: Install the Code
            <p>Install the code with the command:</p>
            <pre><code>pip install -e .</code></pre>

            <div class="note">
                <div class="note-title">Tip</div>
                <p>The <code>-e</code> flag installs the package in editable mode, allowing you to modify the source code without reinstalling.</p>
            </div>

            <h2 id="example">Example</h2>
            <p>Example</p>
            <ul>

            </ul>

            <h2 id="contributing">Contributing</h2>
            <p>Contributors</p>

            <h2 id="license">License</h2>
            <p>Example</p>
            <ul>

            </ul>

            <!-- Footer Navigation -->
            <div class="footer-nav">
                <a href="#overview">← Introduction</a>
                <a href="#inputs">User Guide →</a>
            </div>
        </main>
    </div>
</body>
</html>
