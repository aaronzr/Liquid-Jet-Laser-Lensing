import numpy as np
from collections.abc import Iterable

def n2a(num: int | float | list | np.ndarray):
    # Promote numbers to 1D arrays
    if not isinstance(num, Iterable):
      return np.array([num])
    return np.array(num)

# Construct complex beam parameter `q` from radius `r`, wavelength `l`, 
# index of refraction `n`, and beam width `w`
def q(r, l, n, w):
  return 1/(1/r - 1j * l/(np.pi * n * w**2))

# Extract beam waist from complex q
def q2w(q, n, l):
  return np.real(np.sqrt(-l/(np.pi * n * np.imag(1/q)) + 0j))

# convert between FWHM and beam waist
def f2w(fwhm):
  return fwhm / np.sqrt(2*np.log(2))

def w2f(w):
  return w * np.sqrt(2*np.log(2))
