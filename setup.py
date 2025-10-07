from setuptools import setup, find_packages

setup(
    name="spinvibe",
    version="1.0",
    author="Victor Chang Lee",
    author_email="victor.changlee@northwestern.edu",
    description="Spin-phonon dynamics for molecular qubits in a crystal lattice",
    long_description=open("README.md").read(),
    url="https://victorchanglee.github.io/SpinVibe/",
    packages=["spinvibe"], 
    package_dir={"spinvibe": "src"},            
    install_requires=[
        "numpy",
        "scipy",
        "mpi4py",
        "h5py"
    ],
    entry_points={
        "console_scripts": [
            "spinvibe = spinvibe.run:main",
        ],
    },
    python_requires=">=3.9",
)
