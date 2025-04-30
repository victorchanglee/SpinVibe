import numpy as np
import argparse
import os
import h5py as h5



def extract_g(input_file):
    """Read the input file, extract the second occurrence of g-matrix data, and store in a NumPy array."""
    g_tensor = np.zeros((3, 3), dtype=np.float64)
   
          
    filename = f"{input_file}.property.txt"


    with open(filename, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    # Find the D_Tensor block
    block_start = next((i for i, line in enumerate(lines) 
                    if line.startswith('$CASSCF_G_Tensor')), None)
    if block_start is None:
        raise ValueError("Required $CASSCF_G_Tensor block not found")

    # Find block end (next $ line or EOF)
    block_end = next((i for i, line in enumerate(lines[block_start+1:], block_start+1)
                    if line.startswith('$')), len(lines))

    # Locate &D_RAW within the block
    block_lines = lines[block_start:block_end]
    d_raw_line = next((i for i, line in enumerate(block_lines) if '&G_MATRIX' in line), None)
    if d_raw_line is None:
        raise ValueError("&G_MATRIX section not found within block")

    # Find data lines (skip headers and empty lines)
    data_lines = []
    current_line = d_raw_line + 1  # Start looking after &D_RAW line
    while len(data_lines) < 3 and current_line < len(block_lines):
        line = block_lines[current_line]
        if line and not line.startswith(('&', '$')):
            parts = line.split()
            if len(parts) >= 4:  # Expect [index, val1, val2, val3]
                try:
                    data_lines.append([float(x) for x in parts[1:4]])
                except ValueError:
                    pass
        current_line += 1

    if len(data_lines) != 3:
        raise ValueError(f"Found {len(data_lines)} valid data lines instead of 3")


    
    g_tensor[:, :] = np.array(data_lines)


    return g_tensor

def extract_d(input_file):
    """Read the input file, extract the third Raw D-tensor data, and store in a NumPy array."""

    d_tensor = np.zeros((3, 3), dtype=np.float64)
 
    filename = f"{input_file}.property.txt"

    with open(filename, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    # Find the D_Tensor block
    block_start = next((i for i, line in enumerate(lines) 
                    if line.startswith('$D_Tensor_NEVPT2_2ndOrder')), None)
    if block_start is None:
        raise ValueError("Required $D_Tensor_NEVPT2_2ndOrder block not found")

    # Find block end (next $ line or EOF)
    block_end = next((i for i, line in enumerate(lines[block_start+1:], block_start+1)
                    if line.startswith('$')), len(lines))

    # Locate &D_RAW within the block
    block_lines = lines[block_start:block_end]
    d_raw_line = next((i for i, line in enumerate(block_lines) if '&D_RAW' in line), None)
    if d_raw_line is None:
        raise ValueError("&D_RAW section not found within block")

    # Find data lines (skip headers and empty lines)
    data_lines = []
    current_line = d_raw_line + 1  # Start looking after &D_RAW line
    while len(data_lines) < 3 and current_line < len(block_lines):
        line = block_lines[current_line]
        if line and not line.startswith(('&', '$')):
            parts = line.split()
            if len(parts) >= 4:  # Expect [index, val1, val2, val3]
                try:
                    data_lines.append([float(x) for x in parts[1:4]])
                except ValueError:
                    pass
        current_line += 1

    if len(data_lines) != 3:
        raise ValueError(f"Found {len(data_lines)} valid data lines instead of 3")


    
    d_tensor[:,:] = np.array(data_lines)

                   
    return d_tensor

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract g-matrix data from input files.")
    parser.add_argument("input_name", help="Base name of the input file.")

    args = parser.parse_args()

    # Call the main function
    g_tensor = extract_g(args.input_name)
    d_tensor = extract_d(args.input_name)

    with h5.File(f"{args.input_name}.h5", "w") as f:
        f.create_dataset("g_tensor", data=g_tensor)
        f.create_dataset("d_tensor", data=d_tensor)


