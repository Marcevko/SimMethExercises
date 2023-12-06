"""
TODO
- ex_5: Nachsehen wo ich die berechnung der gemittelten Observablen einfuegen soll
"""

# introduce classes to the students
class Simulation:
    def __init__(self, dt, x, v, box, r_cut, shift, thermostat_temp=None, force_cap=None):
        self.dt = dt
        self.x = x.copy()
        self.v = v.copy()
        self.box = box.copy()
        self.r_cut = r_cut
        self.shift = shift
        
        self.state = {
            'positions': None,
            'velocities': None,
            'forces': None,
            'energies': None,
            'temperatures': None,
            'pressures': None,
        }

        self.n_dims = self.x.shape[0]
        self.n = self.x.shape[1]
        self.f = np.zeros_like(x)

        # both r_ij_matrix and f_ij_matrix are computed in self.forces()
        self.r_ij_matrix = np.zeros((self.n, self.n, self.n_dims))
        self.f_ij_matrix = np.zeros((self.n, self.n, self.n_dims))
        # computed in e_pot_ij_matrix
        self.e_pot_ij_matrix = np.zeros((self.n, self.n))

        # remove maybe later. Do not think that they are necessary at all.
        self.e_pot = 0.0
        self.e_kin = 0.0
        
        self.T = 0.0
        self.P = 0.0

        self.thermostat_temp = thermostat_temp
        self.force_cap = force_cap


    def distances(self):
        self.r_ij_matrix = np.repeat([self.x.transpose()], self.n, axis=0)
        self.r_ij_matrix -= np.transpose(self.r_ij_matrix, axes=[1, 0, 2])
        # minimum image convention
        image_offsets = self.r_ij_matrix.copy()
        for nth_box_component, box_component in enumerate(self.box):
            image_offsets[:, :, nth_box_component] = \
                np.rint(image_offsets[:, :, nth_box_component] / box_component) * box_component
        self.r_ij_matrix -= image_offsets

    def energies(self):
        r = np.linalg.norm(self.r_ij_matrix, axis=2)
        with np.errstate(all='ignore'):
            self.e_pot_ij_matrix = np.where((r != 0.0) & (r < self.r_cut),
                                            4.0 * (np.power(r, -12.) - np.power(r, -6.)) + self.shift, 0.0)

    def forces(self):
        # first update the distance vector matrix, obeying minimum image convention
        self.distances()
        self.f_ij_matrix = self.r_ij_matrix.copy()
        r = np.linalg.norm(self.r_ij_matrix, axis=2)
        with np.errstate(all='ignore'):
            fac = np.where((r != 0.0) & (r < self.r_cut),
                           4.0 * (12.0 * np.power(r, -13.) - 6.0 * np.power(r, -7.)), 0.0)
        for dim in range(self.n_dims):
            with np.errstate(invalid='ignore'):
                self.f_ij_matrix[:, :, dim] *= np.where(r != 0.0, fac / r, 0.0)
        self.f = np.sum(self.f_ij_matrix, axis=0).transpose()

        if self.force_cap:
            self.force_capping()

    def energy(self):
        """Compute and return the energy components of the system."""
        # compute energy matrix
        self.energies()

        self.e_kin = 0.0
        self.e_pot = 0.0

        #TODO
        # - maybe search for better implementation of sums
        # - think about removing self.e_kin and self.e_pot variables, as they are only for one time point

        # compute potential energy of system as sum over all interacting pair energies
        for i in range(self.n):
            self.e_kin += 0.5*np.dot(self.v[:, i], self.v[:, i])
            for j in range(i + 1, self.n):
                self.e_pot += self.e_pot_ij_matrix[i,j]

        return self.e_pot, self.e_kin

    def temperature(self):
        e_kin = 0.0
        for i in range(self.n):
            e_kin += 0.5*np.dot(self.v[:, i], self.v[:, i])

        self.T = 2*e_kin / (self.n_dims * self.n)
        return self.T

    def pressure(self):
        # self.energies()
        surface_area = 2*self.n_dims*self.box[0]**(self.n_dims - 1)
        force_terms = 0.0
        velocity_terms = 0.0
        
        for i in range(self.n):
            velocity_terms += np.dot(self.v[:, i], self.v[:, i])
            for j in range(i + 1, self.n):
                force_terms += np.dot(self.f_ij_matrix[i,j], self.r_ij_matrix[i,j])

        return (velocity_terms + force_terms) / (2*surface_area)
    
    # implementierung verbessern
    def rdf(self):
        distance_to_first_particle = np.linalg.norm(self.r_ij_matrix[0, 1:, :], axis=1)
        hist, bin_edges = np.histogram(distance_to_first_particle, bins=100, range=(0.8, 5.0))
        bin_middlepoints = [(bin_edges[i+1] + bin_edges[i])/2 for i in range(len(bin_edges) - 1)]
        delta_r = bin_middlepoints[1] - bin_middlepoints[0]

        density = self.n / (self.box[0]**(self.n_dims))
        # ring_area = np.array(splitted_string[1][-2]
        #     [(np.pi/density)*( (bin_location + delta_r)**2 - bin_location**2) for bin_location in bin_middlepoints]
        # )
        ring_area = np.array(
            [(np.pi/density) * (2*np.pi*bin_location*delta_r) for bin_location in bin_middlepoints]
        )

        return hist / ring_area
        

    def propagate(self):
        # update positions
        self.x += self.v * self.dt + 0.5 * self.f * self.dt * self.dt

        # half update of the velocity
        self.v += 0.5 * self.f * self.dt

        # compute new forces
        self.forces()
        # we assume that all particles have a mass of unity

        # second half update of the velocity
        self.v += 0.5 * self.f * self.dt
        # use velocity-rescaling
        if self.thermostat_temp:
            self.velocity_rescale(self.thermostat_temp)

    def save_state(self):
        self.state['positions'] = self.x.copy()
        self.state['velocities'] = self.v.copy()
        self.state['forces'] = self.f.copy()            

    def velocity_rescale(self, thermostat_temperature):
        rescaling_factor = thermostat_temperature / self.temperature()
        self.v *= np.sqrt(rescaling_factor)

    def force_capping(self):
        self.f = np.where(self.f > self.force_cap, self.force_cap, self.f)
        self.f = np.where(self.f < - self.force_cap, - self.force_cap, self.f)

def write_checkpoint(state, path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise RuntimeError("Checkpoint file already exists")
    with open(path, 'wb') as fp:
        pickle.dump(state, fp)


if __name__ == "__main__":
    import argparse
    import pickle
    import itertools
    import logging

    import os.path

    import numpy as np
    import scipy.spatial  # todo: probably remove in template
    import tqdm

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'N_per_side',
        type=int,
        help='Number of particles per lattice side.')
    parser.add_argument(
        '--cpt',
        type=str,
        help='Path to checkpoint.')
    parser.add_argument(
        '--thrm',
        type=float,
        help='Temperatures to be used by the thermostat.',
        default=None)
    parser.add_argument(
        '--frc_cap',
        type=float,
        help='Force-Cap that will be used during the systems warmup.',
        default=None)
    args = parser.parse_args()

    np.random.seed(2)

    DT = 0.01
    T_MAX = 500.0
    N_TIME_STEPS = int(T_MAX / DT)

    R_CUT = 2.5
    SHIFT = 0.016316891136

    TEMP = args.thrm
    FORCE_CAP = args.frc_cap

    DIM = 2
    DENSITY = 0.316
    N_PER_SIDE = args.N_per_side
    N_PART = N_PER_SIDE**DIM
    VOLUME = N_PART / DENSITY
    BOX = np.ones(DIM) * VOLUME**(1. / DIM)

    SAMPLING_STRIDE = 3

    if not args.cpt or not os.path.exists(args.cpt):
        logging.info("Starting from scratch.")
        # particle positions
        if FORCE_CAP:
            x = np.multiply( BOX, np.random.random((DIM, N_PART)).T ).T
        else:
            x = np.array(list(itertools.product(np.linspace(0, BOX[0], N_PER_SIDE, endpoint=False),
                                                np.linspace(0, BOX[1], N_PER_SIDE, endpoint=False)))).T

        # random particle velocities
        v = 0.5*(2.0 * np.random.random((DIM, N_PART)) - 1.0)

        positions = []
        forces = []
        energies = []
        pressures = []
        temperatures = []
        rdfs = []
    elif args.cpt and os.path.exists(args.cpt):
        logging.info("Reading state from checkpoint.")
        with open(args.cpt, 'rb') as fp:
            data = pickle.load(fp)

        # start new observables instead of loading from previous (checkpoint) run, only energies are saved for now!
        positions = []
        forces = []
        energies = []
        pressures = []
        temperatures = []
        rdfs = []

        x = data['positions']
        v = data['velocities']
        energies = data['energies']
        pressures = data['pressures']
        temperatures = data['temperatures']

    sim = Simulation(DT, x, v, BOX, R_CUT, SHIFT, TEMP, FORCE_CAP)

    # If checkpoint is used, also the forces have to be reloaded!
    if args.cpt and os.path.exists(args.cpt):
        sim.f = data['forces']

    for i in tqdm.tqdm(range(N_TIME_STEPS)):
        sim.propagate()

        if not FORCE_CAP:
            if i % SAMPLING_STRIDE == 0:
                positions.append(sim.x.copy())
                pressures.append(sim.pressure())
                forces.append(sim.f)
                energies.append(np.sum(sim.energy()))
                temperatures.append(sim.temperature())
                rdfs.append(sim.rdf())
        else:
            if i % SAMPLING_STRIDE == 0:
                forces.append(sim.f)
                smaller_than_cap_force = sim.f < sim.force_cap
                if np.all(smaller_than_cap_force):
                    break
                sim.force_cap *= 1.1

    if args.cpt:
        sim.save_state()
        state = sim.state.copy()
        state['forces_all'] = forces
        state['energies'] = energies
        state['pressures'] = pressures
        state['temperatures'] = temperatures
        state['rdfs'] = rdfs
        write_checkpoint(state, args.cpt, overwrite=True)
