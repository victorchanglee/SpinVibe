import numpy as np
import sys
from mpi4py import MPI
from . import spin_phonon, read_files, math_func


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Input files
    spin_file = 'Cr_otolyl_4.h5'
    phonon_file = 'Sn_otolyl_4_223.h5'
    d1_file = 'Cr_otolyl_4_d1.h5'
    disp1 = np.linspace(-0.05, 0.05, 10)

    d2_file = 'd_predict.h5'
    g2_file = 'g_predict.h5'
    disp2 = np.array([-0.05, -0.0278, -0.0056, 0.0056, 0.0278, 0.05])

    atoms_file = 'Sn_otolyl_4_atoms.h5'
    indices_file = 'molecule_indices.h5'
    mol_mass = 'mol_mass.h5'

    if rank == 0:
        print("Input files to load:")
        print("  Unperturbed g and d tensors: ", spin_file)
        print("  Phonon frequencies and eigenvectors: ", phonon_file)
        print("  First derivatives of d and g tensors: ", d1_file)
        print("  Second derivatives of d and g tensors: ", d2_file, g2_file)
        print("  Atomic positions and reciprocal vectors: ", atoms_file)
        print("  Indices of atoms in the molecule in the crystal: ", indices_file)
        print("  Atomic masses: ", mol_mass)

    file_reader = read_files.Read_files(
        spin_file, phonon_file, d1_file, d2_file, g2_file,
        atoms_file, indices_file, mol_mass, disp1, disp2
    )

    # Input parameters
    B = np.array([0.0, 0.0, 0.410])
    S = 1
    init_type = 'pure'
    R_type = None
    Delta_alpha_q = 1  # Broadening
    T = 5             # Temperature
    Ncells = 50

    rot_mat = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])

    pol = np.array([0, 0, 1])

    tf = 1E-3
    dt = 1E-6
    save_file = f'Spin_phonon_{T}K_{Delta_alpha_q}_{Ncells}_{theta}.h5'

    spin_phonon.spin_phonon(
        B, S, Ncells, Delta_alpha_q, rot_mat, pol,
        T, tf, dt, file_reader, save_file, init_type, R_type
    )

    if rank == 0:
        print("Job completed.")


if __name__ == "__main__":
    main()
