"""
Notes for interpretation of the plots:
    - The third particle in the BOUNCE-less sim doesn't behave like a billiard-ball>
    - Explanation: The attractive force of LJ changes the trajectory of the orange AND red balls. The do not hit each other (like hard spheres)
        but also pull each other (when they have small distances).
    - If two particles hit each other directly, then they behave similar to hard spheres. If the fly by another particle in close distance, they interact
        with the attractive part of the LJ-force.

General Notes:
    - BOUNCE can be tuned with the BOUNCE-parameter (bool).
    - The name of the data-file gets tuned with BOUNCE-parameter.

Questions:
    - What is meant with "Make sure all particles are inside the box at init". 
    --> Maybe it is has something to do with the change of one particle position that we are supposed to make
    - Why should I change the position of one of the particles. They all hit each other (in vanilla config)
"""

import numpy as np

from typing import Tuple

import ex_3_2


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


def apply_bounce_back(x: np.ndarray, v: np.ndarray, box_l: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Write helper
    """
    # print(x.shape)
    for atom_index, atom_position in enumerate(x.T):
        # change the velocities dependent on the side they bounce against
        # if atom_index == 0:
        #     print(f"{atom_index}: {atom_position}")

        if atom_position[0] <= 0.0 or atom_position[0] >= box_l:
            v[0, atom_index] = -v[0, atom_index]

        if atom_position[1] <= 0.0 or atom_position[1] >= box_l:
            v[1, atom_index] = -v[1, atom_index]
    
    return x, v


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    DT = 0.01
    T_MAX = 20.0
    N_TIME_STEPS = int(T_MAX / DT)
    BOX_L = 15.0
    BOUNCE = True

    # running variables
    time = 0.0

    # particle positions
    x = np.zeros((2, 5))
    x[:, 0] = [0.0, 0.0]
    x[:, 1] = [5.0, 0.3]
    x[:, 2] = [8.0, 1.8]
    x[:, 3] = [10.9, 0.3]
    x[:, 4] = [12.0, 7.0]

    # shift positions by one to ensure all particles are inside box at init
    x += 1.0
    print(x, x.shape)

    # particle velocities
    v = np.zeros((2, 5))
    v[:, 0] = [2.0, 0.0]
    v[:, 1] = [0.0, 0.0]
    v[:, 2] = [0.0, 0.0]
    v[:, 3] = [0.0, 0.0]
    v[:, 4] = [0.0, 0.0]

    f = forces(x)

    N_PART = x.shape[1]

    positions = np.full((N_TIME_STEPS, 2, N_PART), np.nan)
    energies = np.full((N_TIME_STEPS), np.nan)

    vtf_filename = 'sm1_worksheet_2/plots/ljbillards_bounce.vtf' if BOUNCE else 'sm1_worksheet_2/plots/ljbillards.vtf'

    with open(vtf_filename, 'w') as vtffile:
        # write the structure of the system into the file:
        # N particles ("atoms") with a radius of 0.5
        vtffile.write(f'atom 0:{N_PART - 1} radius 0.5\n')
        for i in range(N_TIME_STEPS):
            if BOUNCE:
                print(x, x.shape)
                x, v = apply_bounce_back(x, v, BOX_L)
            x, v, f = step_vv(x, v, f, DT)
            time += DT

            positions[i, :2] = x
            energies[i] = total_energy(x, v)

            # write out that a new timestep starts
            vtffile.write('timestep\n')
            # write out the coordinates of the particles
            for p in x.T:
                vtffile.write(f"{p[0]} {p[1]} 0.\n")

    traj = np.array(positions)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(N_PART):
        ax1.plot(positions[:, 0, i], positions[:, 1, i], label='{}'.format(i))
    ax1.set_title('Trajectory')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.set_xlim([0.0, 15.0])
    ax1.set_ylim([0.0, 15.0])
    ax1.legend()

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Total energy")
    ax2.plot(energies)
    ax2.set_title('Total energy')
    plt.show()
