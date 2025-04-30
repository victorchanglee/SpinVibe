import h5py
from ase.io import read
from ase.data import atomic_masses
import numpy as np

def qe_out_to_h5(qe_output_file, output_h5):
    # Read the final structure from the QE output
    atoms = read(qe_output_file, format='espresso-out')

    # Extract atomic data
    symbols = atoms.get_chemical_symbols()
    masses = np.array([atomic_masses[atom.number] for atom in atoms])
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    reciprocal_cell = atoms.get_cell().reciprocal()

    # Save to HDF5
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('lattice_vectors', data=cell)
        f.create_dataset('symbols', data=np.array(symbols, dtype='S'))  # byte strings
        f.create_dataset('masses', data=masses)
        f.create_dataset('positions', data=positions)
        f.create_dataset('reciprocal_vectors', data=reciprocal_cell)
    print(f"Saved {len(symbols)} atoms from QE output to '{output_h5}'.")

# Example usage
qe_out_to_h5('scf.out', 'Sn_otolyl_4_atoms.h5')
