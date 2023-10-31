#!/usr/bin/env python3

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

def force(
        mass: float, 
        gravity: float,
    ) -> np.ndarray:
    """
    Returns the weight force dependent on the mass m and the gravitational accelaration g. 
    """
    return np.array([0.0, -1 * mass * gravity])

def step_euler(
        x: np.ndarray, 
        v: np.ndarray,
        dt: float, 
        mass: float,
        gravity: float, 
        f: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    x += v * dt
    v += (dt / mass) * f

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
    plt.show()
