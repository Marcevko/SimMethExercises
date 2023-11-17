"""
Make notes here:


Still TO-DO:

    - Take a look at VMD and take the screenshots
    - Change force function (search for smallest distance between virtual images of particles for LJ potential beyond boundary)
    - Change potentials in forces to cutoff
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

import ex_3_2


def lj_potential(r_ij: np.ndarray, r_cutoff: float) -> float:
    """
    write helper
    """
    vector_norm = np.linalg.norm(r_ij)
    cutoff_potential_height = ex_3_2.lj_potential(np.array([r_cutoff, 0.0]))
    return 4 * ( (1 / vector_norm**12) - (1 / vector_norm**6) ) - cutoff_potential_height if vector_norm <= r_cutoff else 0.0


def lj_force(r_ij: np.ndarray, r_cutoff: float) -> np.ndarray:
    """
    Write helper
    """
    vector_norm = np.linalg.norm(r_ij)
    return ex_3_2.lj_force(r_ij) if vector_norm <= r_cutoff else np.zeros(len(r_ij))

# This function has to be changed. The LJ pot has to interact with the virtual image of all particles on the boundaries
def forces(x: np.ndarray) -> np.ndarray:
    """Compute and return the forces acting onto the particles,
    depending on the positions x."""
    N = x.shape[1]
    f = np.zeros_like(x)
    for i in range(1, N):
        for j in range(i):
            # distance vector
            r_ij = x[:, j] - x[:, i]
            f_ij = ex_3_2.lj_force(r_ij)
            f[:, i] -= f_ij
            f[:, j] += f_ij
    return f


def total_energy(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute and return the total energy of the system with the
    particles at positions x and velocities v."""
    N = x.shape[1]
    E_pot = 0.0
    E_kin = 0.0
    # sum up potential energies
    for i in range(1, N):
        for j in range(i):
            # distance vector
            r_ij = x[:, j] - x[:, i]
            E_pot += ex_3_2.lj_potential(r_ij)
    # sum up kinetic energy
    for i in range(N):
        E_kin += 0.5 * np.dot(v[:, i], v[:, i])
    return E_pot + E_kin


def step_vv(x: np.ndarray, v: np.ndarray, f: np.ndarray, dt: float):
    # update positions
    x += v * dt + 0.5 * f * dt * dt
    # half update of the velocity
    v += 0.5 * f * dt

    # compute new forces
    f = forces(x)
    # we assume that all particles have a mass of unity

    # second half update of the velocity
    v += 0.5 * f * dt

    return x, v, f

# iteriert falsche axis, siehe 3.3
def apply_pbc(x: np.ndarray, v: np.ndarray, box_l: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Write helper
    """
    for atom_position in x:
        boundary_passed = atom_position // box_l
        
        if boundary_passed[0] != 0:
            atom_position[0] -= boundary_passed[0] * box_l 

        if boundary_passed[1] != 0:
            atom_position[1] -= boundary_passed[1] * box_l 
    
    return x, v


def test_pbc():
    BOX_L = 10.0
    DT = 0.01
    T_MAX = 20.0
    N_TIME_STEPS = int(T_MAX / DT)

    x = np.zeros((2, 2))
    x[:, 0] = [3.9, 3.0]
    x[:, 1] = [6.1, 5.0]

    v = np.zeros((2, 2))
    v[:, 0] = [-2.0, -2.0]
    v[:, 1] = [2.0, 2.0]

    # running variables
    time = 0.0

    f = forces(x)

    N_PART = x.shape[1]

    positions = np.full((N_TIME_STEPS, 2, N_PART), np.nan)
    energies = np.full((N_TIME_STEPS), np.nan)


    for i in range(N_TIME_STEPS):
        x, v = apply_pbc(x, v, BOX_L)
        x, v, f = step_vv(x, v, f, DT)
        time += DT

        positions[i, :2] = x
        energies[i] = total_energy(x, v)

    traj = np.array(positions)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(N_PART):
        ax1.plot(positions[:, 0, i], positions[:, 1, i], 'o', label='{}'.format(i), markersize=2)
    ax1.set_title('Trajectory')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.legend()

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Total energy")
    ax2.plot(energies)
    ax2.set_title('Total energy')
    plt.show()

    return


if __name__ == "__main__":
    # test_pbc()

    # DT = 0.01
    # T_MAX = 20.0
    # N_TIME_STEPS = int(T_MAX / DT)
    # BOX_L = 15.0
    # BOUNCE = True

    # # running variables
    # time = 0.0

    # # particle positions
    # x = np.zeros((2, 5))
    # x[:, 0] = [0.0, 0.0]
    # x[:, 1] = [5.0, 0.3]
    # x[:, 2] = [8.0, 1.8]
    # x[:, 3] = [10.9, 0.3]
    # x[:, 4] = [12.0, 7.0]

    # # particle velocities
    # v = np.zeros((2, 5))
    # v[:, 0] = [2.0, 0.0]
    # v[:, 1] = [0.0, 0.0]
    # v[:, 2] = [0.0, 0.0]
    # v[:, 3] = [0.0, 0.0]
    # v[:, 4] = [0.0, 0.0]

    # f = forces(x)

    # N_PART = x.shape[1]

    # positions = np.full((N_TIME_STEPS, 2, N_PART), np.nan)
    # energies = np.full((N_TIME_STEPS), np.nan)

    # vtf_filename = 'sm1_worksheet_2/plots/ljbillards_bounce.vtf' if BOUNCE else 'sm1_worksheet_2/plots/ljbillards.vtf'

    # with open(vtf_filename, 'w') as vtffile:
    #     # write the structure of the system into the file:
    #     # N particles ("atoms") with a radius of 0.5
    #     vtffile.write(f'atom 0:{N_PART - 1} radius 0.5\n')
    #     for i in range(N_TIME_STEPS):
    #         if BOUNCE:
    #             x, v = apply_bounce_back(x, v, BOX_L)
    #         x, v, f = step_vv(x, v, f, DT)
    #         time += DT

    #         positions[i, :2] = x
    #         energies[i] = total_energy(x, v)

    #         # write out that a new timestep starts
    #         vtffile.write('timestep\n')
    #         # write out the coordinates of the particles
    #         for p in x.T:
    #             vtffile.write(f"{p[0]} {p[1]} 0.\n")

    # traj = np.array(positions)

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # for i in range(N_PART):
    #     ax1.plot(positions[:, 0, i], positions[:, 1, i], label='{}'.format(i))
    # ax1.set_title('Trajectory')
    # ax1.set_aspect('equal')
    # ax1.set_xlabel('x position')
    # ax1.set_ylabel('y position')
    # ax1.legend()

    # ax2.set_xlabel("Time step")
    # ax2.set_ylabel("Total energy")
    # ax2.plot(energies)
    # ax2.set_title('Total energy')
    # plt.show()

    test_pbc()

    distance_vector = np.zeros((1000, 2))
    distance_vector[:, 0] = np.linspace(0.85, 3.0, 1000)

    lj_potential_computed = np.array([ex_3_2.lj_potential(distance) for distance in distance_vector])
    lj_truncated_computed = np.array([lj_potential(distance, 2.5) for distance in distance_vector])

    lj_force_computed = np.array([ex_3_2.lj_force(distance) for distance in distance_vector])
    lj_force_truncated = np.array([lj_force(distance, 2.5) for distance in distance_vector])

    print(lj_force_computed.shape)
    print(lj_force_truncated.shape)

    plt.plot(distance_vector[:, 0], lj_potential_computed, label='default LJ')
    plt.plot(distance_vector[:, 0], lj_truncated_computed, label='truncated LJ')
    plt.legend()
    plt.ylim([-1.2, 1])
    plt.show()

    plt.plot(distance_vector[:, 0], lj_force_computed[:, 0], label='default LJ force')
    plt.plot(distance_vector[:, 0], lj_force_truncated[:, 0], label='truncated LJ force')
    plt.legend()
    plt.ylim([-1.2, 1])
    plt.show()