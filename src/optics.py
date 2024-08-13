'''
optics.py
======
Provides an API for simulating linear/paraxial optical systems
using ray-tracing (ABCD) matrices and propagating Gaussian beams.

Example usage:
[TODO]
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from abc import ABC, abstractmethod
import typing
from typing import List


from .utils import *

# base class
class OpticalComponent():
  '''
  class OpticalComponent
  =====
  An OpticalComponent is an arbitrary optical component with a ray transfer
  matrix `M`. The ambient index of refraction is n0. If the component is an
  Interface, the index of refraction changes at the interface.

  TODO:
  - add Fourier transfer function `h`/`H` field
  '''
  # an arbitrary optical component with ray transfer matrix M
  def __init__(self, M: np.ndarray, n0=1):
    self.M = M
    self.n0 = n0
  def __repr__(self):
    return f'OpticalComponent(M={self.M}, n0={self.n0})'
  def __matmul__(self, other):
    return self.M @ other.M

# unused -- current implementation fills in free space between components
class FreeSpace(OpticalComponent):
  def __init__(self, d):
    self.M = np.array([[1, d], [0, 1]])
    self.d = d
  def __repr__(self):
    return f'FreeSpace(d={self.d}), hellohello'

class ThinLens(OpticalComponent):
  def __init__(self, f):
    self.M = np.array([[1, 0], [-1/f, 1]])
    self.f = f
  def __repr__(self):
    return f'ThinLens(f={self.f})'

class Interface(OpticalComponent):
  def __init__(self, n1, n2):
    self.n1 = n1
    self.n2 = n2

class SphericalInterface(Interface):
  def __init__(self, R, n1, n2):
    super().__init__(n1, n2)
    self.R = R
    self.M = np.array([[1, 0], [(n1-n2)/(R*n2), n1/n2]])
  def __repr__(self):
    return f'SphericalInterface(R={self.R}, n1={self.n1}, n2={self.n2})'

class FlatInterface(SphericalInterface):
  def __init__(self, n1, n2):
    super().__init__(np.infty, n1, n2)
  def __repr__(self):
    return f'FlatInterface(n1={self.n1}, n2={self.n2})'


'''
class OpticalSystem
=====
An OpticalSystem consists of several OpticalComponents located at various
locations along the z-axis. The system's "ambient" index of refraction is n0.
`sys = OpticalSystem(n0=n0)`

You can add an OpticalComponent to the system at location z by calling:
`sys.add(OpticalComponent, z)`

You can also add another OpticalSystem to the system at location z by calling:
`sys.add(OpticalSystem, z)`

In this case, all the components of the OpticalSystem will be added
at the z offset specified.
'''

class OpticalSystem(OpticalComponent):

  def __init__(self, n0=1, *values: tuple[OpticalComponent, float]):
    self.n0 = n0
    # A dictionary mapping z coordinates to optical components
    self.components = {} # dict[np.float64 : OpticalComponent]
    # A dictionary mapping z coordinates to index of refraction (left to right)
    # e.g. {0: 1.3, 1: 1.6} would mean:
    # <---(n=n0)---|---(n=1.3)---|---(n=1.6)--->
    #             z=0           z=1
    self.n_list = {}
    for pair in values:
      self.add(*pair)

  def __repr__(self):
    repr = f'OpticalSystem(n0={self.n0}):'
    for (z, component) in self.components.items():
      repr += f'\n z={z}:' + '\t' + str(component)
    return repr + '\n'
  
  def n(self, z: int | float | list | np.ndarray):
    '''
    Get the index of refraction for a list of locations `z`

    TODO: optimize to make a single pass through self.n_list
    '''
    ret = []
    for z_i in self.n2a(z):
      # get index of refraction at a single location z_i
      # get the locations of the changes in n that happen before z_i
      changes_before_z_i = [z_n for z_n in self.n_list.keys() if z_n <= z_i]
      if changes_before_z_i == []:
        # If there have been no index changes, default to n0
        ret.append(self.n0)
      else:
        # Otherwise, return the most recent index of refraction
        ret.append(self.n_list[changes_before_z_i[-1]])
    # return the index of refraction at location(s) `z`
    return np.array(ret)

  def M_free_space(self, d):
    # free space propagation
    return np.array([[1, d], [0, 1]])

  def get_matrix(self, z: list[float | int] | np.ndarray):
    '''
    `z` is a list of at least two locations: `z = (z1, z2[, ..., z_n])`
    
    Return a stack of `n-1` ray transfer matrices for the part of the system between 
    `z1` and `z2`, `z2` and `z3`, ..., `z_{n-1}` and `z_n`.
    '''
    z = n2a(z)
    assert(len(z.shape) == 1 and len(z) >= 2)
    N_z = len(z)
    # Ensure z points are in the correct order
    z.sort()
    
    z1 = z[0]
    z2 = z[-1]
    # Enumerate components between z1 and z2
    components_in_range = [comp for comp in sorted(self.components.keys()) \
                            if comp >= z1 and comp < z2]
    
    M_list = np.zeros((N_z - 1, 2, 2))
    
    for i in range(1, len(z)):

      # get a sorted list of locations of optical components in [z1, z2)
      # start from z1
      z1 = z[i]
      z2 = z[i-1]
      z_curr = z1
      M = np.eye(2)
      # for each component in [z1, z2):
      for i in range(len(components_in_range)):
        # propagate up to component from z_curr
        d_i = components_in_range[i] - z_curr
        M = FreeSpace(d_i).M @ M
        z_curr = components_in_range[i]
        # apply the component
        M = self.components[components_in_range[i]].M @ M

      # propagate the rest of the way from the last component to z2
      d_final = z2 - z_curr
      M = self.M_free_space(d_final) @ M
      M_list[i-1] = M
    
    return M_list

  def add(self, component: OpticalComponent, z):
    '''
    Add an optical component to the system at location z
    '''
    # If we are adding a system with subcomponents, add them individually
    if issubclass(type(component), OpticalSystem):
      for (z_pos, subcomponent) in component.components.items():
        self.add(subcomponent, z + z_pos)
    else:
      self.components[z] = component
      if issubclass(type(component), Interface):
        self.n_list[z] = component.n2

  def apply(self, matrix, q_value):
    '''
    Apply the ray transfer `matrix` to the `q_value`.

    Params:
      matrix:   ndarray((N_matrix, 2, 2))
      q_value:  ndarray((N_q))
    Returns:
      q_prime:  ndarray((N_q, N_matrix))
    '''
    assert q_value.shape == matrix.shape[:-2] # N_q
    # The "components" way.
    A, B, C, D = matrix[...,0,0], matrix[...,0,1], matrix[...,1,0], matrix[...,1,1] # shape: (N_q)
    return (np.outer(q_value, A) + B) / (np.outer(q_value, C) + D)

  def propagate(self, q0, z):
    '''
    Propagate Gaussian beam(s) `q0` through the optical system. Record q(z)
    at each location in `z`. At z[0], q(z) = q0.

    If `q0` is a list, return a matrix of shape `(len(q0), len(z))`.
    z must have at least 2 elements, a start point and an endpoint.

    Params:
      q0:   ndarray((N_q0))
      z:    ndarray((N_z))
    Returns:
      q_z:  ndarray((N_q0, N_z - 1))
      
    '''
    assert (z == sorted(z)).all()

    q0 = n2a(q0)

    # Get stack of matrices
    M = self.get_matrix(z)  # shape (N_z - 1, 2, 2)
    
    # Pre-allocate q_z
    q_z = np.zeros_like((*q0.shape, M.shape[0]))

    # Fill in each q_z as we propagate
    for i in range(M.shape[0]):
      q_z[:,i] = self.apply(M[i], q)
    
    q_z = np.zeros(q.shape)

    Q_Z = np.array(Q_Z)
    assert(Q_Z.shape[0] == len(q0))
    assert(Q_Z.shape[1] == len(z))
    return np.array(Q_Z)
  
  def min_w(self, q0, z):
    '''
    TODO: find min_z(w) and argmin_z(w)

    Params:
      q0:     ndarray((N_q0))
      z:      ndarray((N_z))
    Returns:
      w_min:  ndarray((N_q0))
      z_min:  ndarray((N_q0))
    '''
    q = self.propagate(q0, z)
    

class ThickLens(OpticalSystem):
  # SphericalInterface + free space + SphericalInterface
  # centered at z=0, interfaces at +/- t/2
  def __init__(self, R1, R2, t, n, n0=1):
    super().__init__(n0=n0)
    self.n0 = n0
    self.add(SphericalInterface(R1, n1=n0, n2=n), z=-t/2)
    self.add(SphericalInterface(R2, n1=n, n2=n0), z=t/2)
    self.R1 = R1
    self.R2 = R2
    self.t = t
    self.n = n
    self.n0 = n0
  def __repr__(self):
    return f'ThickLens(R1={self.R1}, R2={self.R2}, t={self.t}, n={self.n}, n0={self.n0})'

class Sheet(OpticalSystem):
  # curved interface + free space + curved interface
  # centered at z=0, sphere interfaces at +/- t/2
  def __init__(self, t, n, n0=1):
    super().__init__(n0=n0)
    self.n0 = n0
    self.add(FlatInterface(n1=n0, n2=n), z=-t/2)
    self.add(FlatInterface(n1=n, n2=n0), z=t/2)
    self.t = t
    self.n = n
    self.n0 = n0
  def __repr__(self):
    return f'Sheet(t={self.t}, n={self.n}, n0={self.n0})'