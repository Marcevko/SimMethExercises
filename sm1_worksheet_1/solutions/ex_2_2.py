#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import ex_2_1

def force(
        mass: float, 
        gravity: float,
        v: np.ndarray,
        gamma: float, 
        v_0: np.ndarray
    ) -> np.ndarray:
    """
    Computes the force acting on the particle, including wind friction.

    Args:
        mass: mass of the particle
        gravity: gravitational accleration constant
        v: velocity of the particle
        gamma: friction parameter
        v_0: wind velocity
    """
    
    wind_friction = -gamma*( v - v_0 )
    return ex_2_1.force(mass, gravity) + wind_friction

def run(
        x: np.ndarray,
        v: np.ndarray, 
        dt: float, 
        mass: float, 
        gravity: float, 
        gamma: float, 
        v_0: np.ndarray
    ) -> np.ndarray:
    """
    Runs the simulation while-loop for a set of parameters.

    Args:
        x: starting position vector of the particle
        v: starting velocity vector of the particle
        dt: time-step size per integration step
        mass: mass of the particle
        gravity: gravitational accleration constant
        gamma: friction parameter
        v_0: wind velocity in x-direction
    
    Returns:
        trajectory (np.ndarray): Trajectory of the particle
    """
    x_past, v_past = x.copy(), v.copy()
    x_t, v_t = np.zeros(2), np.zeros(2)
    trajectory = [x_past.copy()]

    wind_vector = np.array([v_0, 0.0])

    running = True
    
    while running:
        x_t, v_t = ex_2_1.step_euler(
            x_past,
            v_past,
            dt,
            mass,
            gravity,
            force(mass, gravity, v, gamma, wind_vector),
        )

        if x_t[1] < 0.0:
            running = False
            break
        
        trajectory.append(x_t.copy())
        x_past, v_past = x_t.copy(), v_t.copy()


    return np.array(trajectory)


if __name__ == "__main__":
    static_args = [
        np.array([0.0, 0.0]),
        np.array([50.0, 50.0]),
        0.1,
        2.0,
        9.81,
        0.1,
    ]
    
    # iwie ueberlegen wie ich an die Traj komme

    # trajectory_no_friction = 
    trajectory_no_wind = run(*static_args, 0.0)
    trajectory_wind = run(*static_args, -50.0)

    print(trajectory_wind)

    plt.plot(trajectory_no_wind[:, 0], trajectory_no_wind[:, 1], label='no wind')
    plt.plot(trajectory_wind[:, 0], trajectory_wind[:, 1], label='wind')
    plt.legend()
    plt.show()