import numpy as np
import numba as nb


real = nb.float64
real_np = np.float64


@nb.njit(real[:](real, real, real), inline="always")
def vector(x: real_np, y: real_np, z: real_np) -> np.ndarray:
    return np.array([x, y, z])

ray = vector
color = vector


@nb.njit(real(real[:]), inline="always")
def length_squared(vector: np.ndarray) -> real_np:
    return vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]


@nb.njit(real(real[:]), inline="always")
def length(vector: np.ndarray) -> real_np:
    return np.sqrt(length_squared(vector))


@nb.njit(real(real[:], real[:]), inline="always")
def dot(u: np.ndarray, v: np.ndarray) -> real_np:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@nb.njit(real[:](real[:], real[:]), inline="always")
def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )


@nb.njit(real[:](real[:]), inline="always")
def unit_nvector(vector: np.ndarray) -> np.ndarray:
    return vector / length(vector)


