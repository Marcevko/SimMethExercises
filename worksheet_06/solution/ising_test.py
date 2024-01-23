#!/usr/bin/env python3

import cising
import numpy as np


def compute_energy(sigma):
    L = sigma.shape[0]
    shifted_ixs = np.roll(np.arange(0, L), -1)
    E = -(sigma*sigma[shifted_ixs, :]).sum()
    E -= (sigma*sigma[:, shifted_ixs]).sum()
    return E


beta = 1./5.
l = 4
assert l >= 4
I = cising.IsingModel(beta, l)
I.set(3, 2, 1)
assert I.get(3, 2) == 1
I.set(3, 2, -1)
assert I.get(3, 2) == -1
I_numpy = I.as_numpy()
assert I_numpy.shape == (4, 4)

assert np.all(np.abs(I_numpy.flatten()) == 1)
assert I.magnetization() == np.average(I_numpy)


for v in -1, 1:
    for i in range(5):
        for j in range(5):
            I.set(i, j, v)
    assert I.magnetization() == v
    assert I.energy() == -2. * l*l
for i in range(5):
    for j in range(5):
        I.set(i, j, np.random.randint(0, 1)*2-1)
        assert I.energy() == compute_energy(I.as_numpy())


for i in range(1000):
    I.try_random_flip()
    assert I.energy() == compute_energy(I.as_numpy())


# Validate against brute force solution
Es = []
Ms = []
for i in range(100000):
    I.try_many_random_flips(500)
    Es.append(I.energy())
    Ms.append(I.magnetization())

assert abs(np.average(Es)/(l*l) + 0.4561353695) < 0.01

assert abs(np.average(np.abs(Ms)) - 0.342765627554) < 0.001
