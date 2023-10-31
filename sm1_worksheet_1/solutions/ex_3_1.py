#!/usr/bin/env python3

import sys
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import scipy.constants

def force(
        r_ij: np.ndarray, 
        m_i: float, 
        m_j: float,
        g: float
    ) -> np.ndarray:
    """
    Computes the gravitational force between two planets.
    """
    r_ij_abs = np.linalg.norm(r_ij)
    return -g * m_i * m_j * r_ij / (r_ij_abs)**3

def step_euler(x, v, dt, mass, g, forces):
    pass

def forces(
        x: np.ndarray,
        masses: np.ndarray,
        g: float
    ) -> np.ndarray:
    """
    Computes the commulative acting forces on all planets as the sum of all smaller forces from 'force'.    
    
    Returns:
    np.ndarray (2)
    """
    pass

if __name__ == "__main__":
    # load the npz file
    try:
        solar_system_data = np.load('files/solar_system.npz')
    except Exception as e:
        logging.warning(e)
        # implement general way of loading the path
        # os.chdir oder sys.path.append
        
    names = solar_system_data['names']
    x_init = solar_system_data['x_init']
    v_init = solar_system_data['v_init']
    m = solar_system_data['m']
    g = solar_system_data['g']

    print(x_init, x_init.shape)
