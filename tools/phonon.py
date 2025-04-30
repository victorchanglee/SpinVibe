import h5py
import numpy as np

def read_quantum_espresso_phonon(filename):
    """
    Reads phonon frequencies and eigenvectors from a Quantum ESPRESSO output file.

    Args:
        filename (str): Path to the output file.

    Returns:
        list: A list of dictionaries, each containing:
            - 'q_point': List of three floats representing the q-point.
            - 'modes': List of dictionaries, each with:
                - 'frequency_THz': Frequency in THz.
                - 'frequency_cm-1': Frequency in cm⁻¹.
                - 'eigenvector': List of complex numbers representing the eigenvector.
    """
    data = []
    current_q = None
    current_modes = []
    current_mode = None

    with open(filename, 'r') as f:
        for line in f:
            stripped_line = line.strip()

            # Check for q-point line
            if line.startswith(' q ='):
                # Save previous q-point data if exists
                if current_q is not None:
                    data.append({
                        'q_point': current_q,
                        'modes': current_modes
                    })
                # Parse new q-point
                q_part = line.split('=')[1].strip()
                q_components = list(map(float, q_part.split()))
                if len(q_components) != 3:
                    raise ValueError(f"Invalid q-point format: {line.strip()}")
                current_q = q_components
                current_modes = []
                current_mode = None

            # Check for frequency line
            elif stripped_line.startswith('freq ('):
                parts = stripped_line.split('=')
                if len(parts) < 3:
                    raise ValueError(f"Invalid frequency line: {stripped_line}")
                # Extract THz and cm⁻¹ values
                cm1_str = parts[2].split('[')[0].strip()
                try: 
                    cm1 = float(cm1_str)
                except ValueError:
                    raise ValueError(f"Could not parse frequencies in line: {stripped_line}")
                # Create new mode entry
                current_mode = {
                    'frequency_cm-1': cm1,
                    'eigenvector': []
                }
                current_modes.append(current_mode)

            # Check for eigenvector line
            elif stripped_line.startswith('('):
                if current_mode is None:
                    continue  # Skip if not inside a mode
                content = stripped_line[1:-1].strip()  # Remove parentheses
                numbers = content.split()
                if len(numbers) != 6:
                    raise ValueError(
                        f"Eigenvector line must have 6 elements, found {len(numbers)}: {stripped_line}"
                    )
                # Parse complex numbers
                complex_parts = []
                for i in range(0, 6, 2):
                    real = float(numbers[i])
                    imag = float(numbers[i+1])
                    complex_parts.append(complex(real, imag))
                current_mode['eigenvector'].extend(complex_parts)

        # Add the last q-point after file ends
        if current_q is not None:
            data.append({
                'q_point': current_q,
                'modes': current_modes
            })

    return data
def save_phonon_h5(input_file, output_file):
    """
    Reads phonon data from Quantum ESPRESSO output and saves to HDF5 format.
    
    Args:
        input_file (str): Path to Quantum ESPRESSO output file
        output_file (str): Path to output HDF5 file
    """
    data = read_quantum_espresso_phonon(input_file)
    
    all_q_points = []
    
    all_freq_cm = []
    all_eigenvectors = []
    
    for q_entry in data:
        q_point = q_entry['q_point']
        modes = q_entry['modes']
        
        all_q_points.append(q_point)
        freq_cm = []
        eigenvectors = []
        
        for mode in modes:
            freq_cm.append(mode['frequency_cm-1'])
            
            # Convert to complex128 explicitly
            ev_flat = np.array(mode['eigenvector'], dtype=np.complex128)
            num_atoms = len(ev_flat) // 3
            ev_reshaped = ev_flat.reshape(num_atoms, 3)
            eigenvectors.append(ev_reshaped)
     
        all_freq_cm.append(freq_cm)
        all_eigenvectors.append(eigenvectors)
    
    # Convert to numpy arrays with explicit typing
    q_array = np.array(all_q_points, dtype=np.float64)
    freq_cm_array = np.array(all_freq_cm, dtype=np.float64)
    eigen_array = np.array(all_eigenvectors, dtype=np.complex128)
    
    # Get dimensions correctly
    num_q, num_modes, num_atoms, num_components = eigen_array.shape
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('q_points', data=q_array)
        f.create_dataset('frequencies_cm', data=freq_cm_array)
        f.create_dataset('eigenvectors', data=eigen_array)
        
        # Add shape information as attributes
        f.attrs['num_q'] = num_q
        f.attrs['num_modes'] = num_modes
        f.attrs['num_atoms'] = num_atoms

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input phonon file')
    parser.add_argument('output', help='Output HDF5 file')
    args = parser.parse_args()
    
    save_phonon_h5(args.input, args.output)
    
