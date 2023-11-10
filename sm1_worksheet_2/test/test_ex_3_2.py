#!/usr/bin/env python3

import sys
import pathlib

import numpy as np
import unittest as ut
import scipy.linalg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.joinpath('solutions')))
import ex_3_2 # isort:skip

class Tests(ut.TestCase):
    r_ij = np.array([1.1, 1.2, -1.3])
    r = scipy.linalg.norm(r_ij)
    
    def test_lj_potential(self):
        self.assertEqual(ex_3_2.lj_potential([1, 0, 0]), 0.0)
        self.assertAlmostEqual(ex_3_2.lj_potential([2**(1./6), 0, 0]), -1.0)

    def test_lj_force(self):
        eps = sys.float_info.epsilon
        self.assertLess(ex_3_2.lj_force(np.array([2**(1./6) + eps, 0, 0]))[0], 0.0)
        self.assertGreater(ex_3_2.lj_force(np.array([2**(1./6) - eps, 0, 0]))[0], 0.0)


if __name__ == "__main__":
    ut.main()
