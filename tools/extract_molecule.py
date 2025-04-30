import numpy as np
from ase.io import read, write
from ase.data import covalent_radii
from scipy.sparse.csgraph import connected_components
import h5py

def isolate_largest_molecule(input_file, output_file, bond_factor=1.2):
    # Read the input CIF file
    atoms = read(input_file)
    
    # Calculate covalent radii for each atom
    radii = covalent_radii[atoms.numbers]
    
    # Calculate distance matrix (without periodic boundary conditions)
    distance_matrix = atoms.get_all_distances(mic=False)
    
    # Create connectivity matrix
    threshold = (radii[:, None] + radii) * bond_factor
    connectivity = distance_matrix < threshold
    np.fill_diagonal(connectivity, False)
    
    # Find connected components (molecules)
    n_components, labels = connected_components(connectivity, directed=False)
    
    # Find the largest molecule
    unique, counts = np.unique(labels, return_counts=True)
    largest_idx = np.argmax(counts)
    mask = (labels == unique[largest_idx])
   
    # Get atom indices of the largest molecule
    molecule_indices = np.where(mask)[0]


    
    # Print indices
    print(f"Atom indices in largest molecule (0-based): {', '.join(map(str, molecule_indices))}")

    # Extract the largest molecule
    largest_molecule = atoms[mask]
    symbols = atoms.get_chemical_symbols()
    symbols_array = np.array(symbols, dtype='S2')

    # Rebuild a local connectivity matrix just for the extracted molecule
    sub_distance_matrix = largest_molecule.get_all_distances(mic=False)
    sub_radii = covalent_radii[largest_molecule.numbers]
    sub_threshold = (sub_radii[:, None] + sub_radii) * bond_factor
    sub_connectivity = sub_distance_matrix < sub_threshold
    np.fill_diagonal(sub_connectivity, False)



    # Save to HDF5 (.h5) file
    list_output_file = output_file.replace('.cif', '_indices.h5')
    with h5py.File(list_output_file, 'w') as f:
        f.create_dataset('indices', data=molecule_indices)
        f.create_dataset('atom_symbols', data=symbols_array)
        
    print(f"Saved atom indices pairs to {list_output_file}")

    # Write to new CIF file
    write(output_file, largest_molecule)
    print(f"Largest molecule with {len(largest_molecule)} atoms written to {output_file}")

    return largest_molecule

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Isolate the largest molecule from a CIF file')
    parser.add_argument('input', help='Input CIF file')
    parser.add_argument('output', help='Output CIF file')
    parser.add_argument('--bond_factor', type=float, default=1.2,
                       help='Bond factor for connectivity determination (default: 1.2)')
    args = parser.parse_args()
    
    largest_molecule = isolate_largest_molecule(args.input, args.output, args.bond_factor)
