"""Module for defining distance metrics on the torus manifolds."""
import numpy as np


def wrapped_angle_diff(a, b, period=2 * np.pi):
    """
    Since theta_i lives on the circle (is periodic) we need to calculate the angular distance.  
    """
    return (a - b + period / 2) % period - period / 2