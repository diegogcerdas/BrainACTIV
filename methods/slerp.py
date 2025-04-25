import numpy as np


def slerp(v0, v1, num, t0=0, t1=1, DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation between two vectors
    Adapted from: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    """
    def interpolate(t, v0, v1):
        # Copy the vectors to reuse them later
        v0_copy = np.copy(v0)
        v1_copy = np.copy(v1)
        # Normalize the vectors to get the directions and angles
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)
        # Dot product with the normalized vectors (can't use np.dot in W)
        dot = np.sum(v0 * v1)
        # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
        if np.abs(dot) > DOT_THRESHOLD:
            return (1 - t) * v0 + t * v1
        # Calculate initial angle between v0 and v1
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        # Angle at timestep t
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        # Finish the slerp algorithm
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0_copy + s1 * v1_copy
        return v2
    # Create the timesteps
    t = np.linspace(t0, t1, num)
    res = np.array([interpolate(t[i], v0, v1) for i in range(num)])
    return res