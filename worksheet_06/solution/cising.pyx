import numpy as np
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared
from libcpp cimport bool

# declare the interface to the C code
cdef extern from "ising.hpp":
    cppclass Ising:
        Ising(double beta, int l)
        int get(int i, int j)
        void set(int i, int j, int v)
        double get_energy()
        double get_magnetization()
        vector[int] get_data()
        bool try_random_flip()
        void try_many_random_flips(int n)


cdef class IsingModel:
  cdef shared_ptr[Ising] c_ising
  cdef int l

  def __init__(self,double beta, int l):
    self.l = l
    self.c_ising = make_shared[Ising](beta, l)

  def as_numpy(self):
    return np.array(self.c_ising.get().get_data()).reshape((self.l, self.l))

  def get(self, i, j):
    return self.c_ising.get().get(i, j)

  def set(self, i, j, v):
    self.c_ising.get().set(i, j, v)

  def energy(self):
    return self.c_ising.get().get_energy()

  def magnetization(self):
    return self.c_ising.get().get_magnetization()

  def try_random_flip(self):
    self.c_ising.get().try_random_flip()

  def try_many_random_flips(self, n):
    self.c_ising.get().try_many_random_flips(n)
