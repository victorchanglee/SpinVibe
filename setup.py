from setuptools import setup, find_packages

setup(
    name="spinvibe",
    version="1.0",
    author="Victor Chang Lee",
    author_email="victor.changlee@northwestern.edu",
    description="Spin-phonon dynamics for molecular qubits in a crystal lattice",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://victorchanglee.github.io/SpinVibe/",
    packages=["spinvibe"], 
    package_dir={"spinvibe": "src"},
    install_requires=[
        "numpy==1.26.4",
        "scipy==1.13.1",
        "mpi4py==4.0.2",
        "h5py==3.12.1",
    ],
    entry_points={
        "console_scripts": [
            "spinvibe = spinvibe.run:main",
        ],
    },
    python_requires=="==3.9.*",
)
