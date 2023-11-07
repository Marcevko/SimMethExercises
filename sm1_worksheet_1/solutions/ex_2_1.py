#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Callable


def force(
        mass: float, 
        gravity: float,
    ) -> np.ndarray:
    """
    Computes the gravitational force acting on the particle.

    Args:
        mass: mass of the particle
        gravity: gravitational accleration constant

    Returns:
        force (np.ndarray): two-dimension force vector acting on particle
    """
    return np.array([0.0, -1 * mass * gravity])


def step_euler(
        x: np.ndarray, 
        v: np.ndarray,
        dt: float, 
        mass: float,
        gravity: float, 
        f: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes one integration step with the non-symplectic Euler-Algorithm.

    Args:
        x: Two-dimensional position vector of particle -> (2, )
        y: Two-dimensional velocity vector of particle -> (2, )
        dt: time-step size
        mass: mass of particle
        gravity: gravitational force constant
        force: force-vector acting on particle

    Returns:
        Tuple[np.ndarray, np.ndarray] -> [(N, 2), (N, 2)]: updated positions and velocities of all planets
    """
    a = f/ mass
    
    x += v * dt
    v += a * dt

    return x, v


if __name__ == "__main__":
    # init starting parameters
    x0, v0 = np.array([0.0, 0.0]), np.array([50.0, 50.0])
    m0 = 2.0
    dt = 0.1
    g = 9.81

    x_past, v_past = x0.copy(), v0.copy()
    x_t, v_t = np.zeros(2), np.zeros(2)
    
    trajectory = []
    trajectory.append(x_past.copy())

    running = True

    while running:
        x_t, v_t = step_euler(
            x_past, v_past, dt, m0, g, force(m0, g),
        )

        if x_t[1] < 0.0:
            running = False
            break
            
        trajectory.append(x_t.copy())
        x_past, v_past = x_t.copy(), v_t.copy()

    trajectory = np.array(trajectory)

    plt.plot(trajectory[:, 0], trajectory[:, 1], label='numeric simulation')
    plt.legend()
    plt.title(f'Particle trajectory without wind.')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.savefig('plots/particle_trajectory_no_wind.png', format='png', dpi=600)
    plt.show()
