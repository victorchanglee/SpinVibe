import numpy as np
import json
from mpi4py import MPI
from . import spin_phonon, read_files, math_func


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Parse command-line argument for input file
    import sys
    if len(sys.argv) < 2:
        if rank == 0:
            print("Usage: spinvibe-run <input.json>")
        sys.exit(1)

    input_file = sys.argv[1]

    if rank == 0:
        print(f"Reading input parameters from: {input_file}")

    with open(input_file, "r") as f:
        params = json.load(f)

    # Convert lists to numpy arrays where needed
    B = np.array(params["B"])
    disp1 = np.array(params["disp1"])
    disp2 = np.array(params["disp2"])
    rot_mat = np.array(params["rotation_matrix"])
    pol = np.array(params["polarization_axis"])
    supercell = np.array(params["supercell"])
    if rank == 0:
        print("Initializing input files...")
        for key in [
            "spin_file", "phonon_file", "d1_file",
            "d2_file", "g2_file", "atoms_file",
            "indices_file", "mol_mass"
        ]:
            print(f"  {key}: {params[key]}")

    # Create file reader
    file_reader = read_files.Read_files(
        params["spin_file"], params["phonon_file"], params["d1_file"],
        params["d2_file"], params["g2_file"],
        params["atoms_file"], params["indices_file"],
        params["mol_mass"], disp1, disp2
    )

    # Run simulation
    spin_phonon.spin_phonon(
        B=B,
        S=params["S"],
        supercell=supercell,
        Delta_alpha_q=params["Delta_alpha_q"],
        rot_mat=rot_mat,
        pol=pol,
        T=params["T"],
        tf=params["tf"],
        dt=params["dt"],
        file_reader=file_reader,
        save_file=params["save_file"],
        init_type=params["init_type"],
        R_type=params["R_type"]
    )

    if rank == 0:
        print(f"Job completed. Results saved to {params['save_file']}")

if __name__ == "__main__":
    main()
