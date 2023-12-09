import numpy as np
import scipy.linalg

def SC_lattice(N: int, box: np.ndarray) -> np.ndarray:
    """Get an array of positions for N particles in a simple-cubic
        lattice.

        Parameters
        ----------
        N : int
            Number of particles.
        box : (3, 0) array_like
            Simulation box dimentions.

        Returns
        -------
        x : (N, 3) array_like
            Particle positions.

    """
    # INSERT CODE HERE
                

def minimum_image_vector(r1: np.ndarray, r2: np.ndarray, box: np.ndarray):
        """ Get connecting vector of two particles, respecting minimum image convention.

        Parameters
        ----------
        r1 : (3, ) array_like
            Position of particle 1.
        r2 : (3, ) array_like
            Position of particle 2.
        box : (3, ) array_like
            Simulation box dimensions.

        Returns
        -------
        array_like
            Connecting vector for `r1` and `r2` with minimum image convention.

        """
        r_12 = r2 - r1
        r_12 -= np.rint(r_12 / box) * box
        return r_12

def get_verlet_list(x: np.ndarray, box: np.ndarray, r_cut: int, skin: int):
        """Create a list of index pairs of interacting particles.

        Parameters
        ----------
        x : (N, 3) array_like
            Particle positions.
        box : (3, 0) array_like
            Simulation box dimentions.
        r_cut : int
            Cutoff radius.
        skin : int
            Skin (Verlet list).

        Returns
        -------
        x : (N, 3) array_like
            Particle positions at the creation of Verlet list.
        array_like : (, 2)
            Verlet list.
        """
        N = x.shape[0]
        verlet_list = []

        for i in range(1, N):
            for j in range(i):
                r_ij = minimum_image_vector(x[i, :], x[j, :], box)

                if np.linalg.norm(r_ij) < r_cut + skin:
                    verlet_list.append((i, j))

        return np.copy(x), np.array(verlet_list)

def lj_force(r_ij: np.ndarray, r_cut: int):
    """Get forces for two LJ-interacting particles.

        Parameters
        ----------
        r_ij : (3) array_like
            Distance vector between particles i and j.
        r_cut : int
            Cutoff radius.

        Returns
        -------
        array_like : (3)
            Force vector.
    """
    def _lj_force(r_ij):
        r = scipy.linalg.norm(r_ij)
        fac = 4.0 * (12.0 * np.power(r, -13.) - 6.0 * np.power(r, -7.))
        return fac * r_ij / r

    r = scipy.linalg.norm(r_ij)
    if r < r_cut:
        return _lj_force(r_ij)
    return np.zeros_like(r_ij)

def lj_potential(r_ij: np.ndarray, r_cut: int, shift: int):
    """Get LJ potential between two particles, with 
    epsilon = 1 and sigma =1.

        Parameters
        ----------
        r_ij : (3) array_like
            Distance vector between particles i and j.
        r_cut : int
            Cutoff radius.
        shift : int
            shift of the potential.

        Returns
        -------
        int
            Potential value.
    """
    def _lj_potential(r_ij):
        r = scipy.linalg.norm(r_ij)
        return 4.0 * (np.power(r, -12.) - np.power(r, -6.))

    r = scipy.linalg.norm(r_ij)
    if r < r_cut:
        return _lj_potential(r_ij) + shift
    return 0.0

def forces(x: np.ndarray, r_cut: int, verlet_list: np.ndarray, box: np.ndarray):
    """Compute and return the forces acting on the particles,
    depending on the positions x.
    
        Parameters
        ----------
        x : (N, 3) array_like
            Particle positions.
        r_cut : int
            Cutoff radius.
        verlet_list : (N, 2) array_like
            Verlet list.
        box : (3, 0) array_like
            Simulation box dimentions.

        Returns
        -------
        array_like : (N, 3)
            Forces.
    """
    f = np.zeros_like(x)
    
    # INSERT CODE HERE

    return f
