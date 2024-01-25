#ifndef ISING_HPP
#define ISING_HPP

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

// Negative number proof modulo
template <typename T> T mod(T a, T b) {
  T ret = a % b;
  while (ret < 0) {
    ret += b;
  }
  return ret;
}

class Ising {
public:
  Ising(double beta, int l) {
    m_rng.seed(42);

    m_beta = beta;
    m_l = l;

    m_data.resize(l * l);

    for (auto &i : m_data) {
      // TODO
      // YOUR CODE HERE: fill the Ising model by randomly setting
      // each site to +1 or -1
      i = 2 * random_int(2) - 1;

      assert((i == 1) or (i == -1));
    };

    update();
  };

  // Recalculate the magnetization from the state of the Ising model
  void recalculate_magnetization() {
    // TODO
    // Calculate the magnetization of the system and store it in m_M
    m_M = 0.0;

    for (int i=0; i<m_l; i++) {
      for (int j=0; j<m_l; j++) {
        m_M += get(i, j);
      };
    };

    assert((m_M >= -m_l * m_l) and (m_M <= m_l * m_l));
  };

  // Recalculate the energy from the state of the Ising model
  double calculate_energy_pair(int i, int j) {
    double e_ij;

    e_ij = -get(i, j) * (get(i-1, j) + get(i+1, j) + get(i, j+1) + get(i, j-1));

    return e_ij;
  }
  
  void recalculate_energy() {
    // TODO
    // Calculate the energy of the Ising model
    // and store it in m_E
    m_E = 0.0;
    
    for (int i=0; i<m_l; i++) {
      for (int j=0; j<m_l; j++) {
        m_E += calculate_energy_pair(i, j);
      };
    };
    m_E /= 2;
  };

  // Update running values for energy and magnetization
  void update() {
    recalculate_energy();
    recalculate_magnetization();
    m_invalid = false;
  };

  // Get spin at i,j
  int get(int i, int j) {
    int _i = mod(i, m_l);
    int _j = mod(j, m_l);
    assert(_i >= 0 and _i < m_l);
    assert(_j >= 0 and _j < m_l);
    return m_data[get_linear_index(_i, _j)];
  };

  // Set spin at i,j
  void set(int i, int j, int v) {
    assert((v == 1) or (v == -1));
    int _i = mod(i, m_l);
    int _j = mod(j, m_l);
    assert(_i >= 0 and _i < m_l);
    assert(_j >= 0 and _j < m_l);
    m_data[get_linear_index(_i, _j)] = v;
    m_invalid = true;
  };

  // Try to flip a spin and return true if move accepted,
  // update magnetization/energy if move was accepted
  bool try_flip(int i, int j) {
    // TODO 
    // perform a single trial move at position i,j.
    // Decide based on the metropolis criterion whether to accept the flip
    // If it is accepted, update the model in m_data and update the
    // running values for magnetization and energy (m_M and m_E)
    // Return true or false depending on whether the move is accepted
    double delta_e_ij = 2 * get(i, j) * (get(i-1, j) + get(i+1, j) + get(i, j+1) + get(i, j-1));
    double r = random_double();
            
    if (delta_e_ij <= 0.0) {
      set(i, j, -get(i, j));
      return true;
    } else {
      if (r < std::exp(-m_beta * delta_e_ij)) { // vllt funkt das so nicht, sondern braucht variable fuer exp
        set(i, j, -get(i, j));
        return true;
      } else {
        return false;
      }
    }

  };

  // Try flipping a random spin. Return true if move accepted
  // If move was accepted, also update energy and magnetization
  bool try_random_flip() {
    // TODO 
    // Try a single flip on a randomly chosen
    // spin. Re-use (don't copy) existing code.
    // Return true/false depending on whether the move was accepted
    int random_i = random_int(m_l);
    int random_j = random_int(m_l);

    return try_flip(random_i, random_j);
  };

  void try_many_random_flips(int n) {
    // TODO
    // Try n moves at randomly chosen spins.
    // Re-use (don't copy) existing code!
    bool break_condition;
    for (int i=0; i<n; i++) {
      break_condition = try_random_flip();

      // if (break_condition) {
      //   break;
      // }
    }
  }

  // Get the current energy
  double get_energy() {
    if (m_invalid) {
      update();
    }
    return m_E;
  };

  // Get the current magnetization
  double get_magnetization() {
    if (m_invalid) {
      update();
    }
    return m_M / (m_l * m_l);
  };

  // Get the raw data
  std::vector<int> get_data() { return m_data; };

private:
  // Parameters
  double m_beta;
  int m_l;

  // Running magnetization, energy
  double m_M;
  double m_E;
  // Are the running values invalid
  bool m_invalid;

  // Ising model
  std::vector<int> m_data;

  // Random number generator
  std::mt19937 m_rng;
  std::uniform_real_distribution<double> m_uniform_dist;
  double random_double() { return m_uniform_dist(m_rng); }

  int random_int(int below) {
    auto x = random_double();
    auto v = static_cast<int>(x * below);
    assert(v >= 0 and v < below);
    return v;
  };

  // Convert 2d indices from 0<=i,j<l to linear index 0<ind<l*l
  int get_linear_index(int i, int j) {
    // TODO
    // Convert the 2d index (i,j) to a linear index ind
    // the m_data vector, such tht the mapping is unique.

    int index = j*m_l + i;

    assert((index < m_l * m_l) and (index >= 0));
    return index;
  };
};

#endif
