import sys
import h5py
import numpy as np
from ase.io import read

def xyz_to_h5_ase(xyz_file, h5_file):
    # Read XYZ file using ASE
    atoms = read(xyz_file)
    
    # Get atomic properties
    masses = atoms.get_masses()
    symbols = atoms.get_chemical_symbols()
    
    # Convert symbols to numpy array of fixed-length strings
    symbols_array = np.array(symbols, dtype='S2')
    
    # Write to HDF5
    with h5py.File(h5_file, 'w') as hf:
        hf.create_dataset('atomic_masses', data=masses)
        hf.create_dataset('atom_symbols', data=symbols_array)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input.xyz> <output.h5>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    xyz_to_h5_ase(input_file, output_file)
    print(f"Successfully saved atomic data to {output_file}")