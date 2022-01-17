import numpy as np
import numba as nb
from numba.experimental import jitclass


real = nb.float64
Real = np.float64

Vector = np.ndarray
Color = np.ndarray
Point = np.ndarray


@nb.njit(real[:](real, real, real), inline="always", cache=True)
def vector(x: Real, y: Real, z: Real) -> Vector:
    return np.array([x, y, z], dtype=Real)


color = vector
point = vector

WHITE = color(1, 1, 1)
BLUE = color(0.5, 0.7, 1)


@nb.njit(real(real[:]), inline="always", cache=True)
def length_squared(vector: Vector) -> Real:
    return vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]


@nb.njit(real(real[:]), inline="always", cache=True)
def length(vector: Vector) -> Real:
    return np.sqrt(length_squared(vector))


@nb.njit(real(real[:], real[:]), inline="always", cache=True)
def dot(u: Vector, v: Vector) -> Real:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@nb.njit(real[:](real[:], real[:]), inline="always", cache=True)
def cross(u: Vector, v: Vector) -> Vector:
    return vector(
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


@nb.njit(real[:](real[:]), inline="always", parallel=True, cache=True)
def unit_vector(vector: Vector) -> Vector:
    return vector / length(vector)


@nb.njit(real[:](real[:], real[:], real), inline="always", parallel=True, cache=True)
def vector_at_t(origin: Point, direction: Vector, t: Real) -> Vector:
    return origin + direction * t


@nb.njit(
    real(real[:], real, real[:], real[:]), inline="always", parallel=True, cache=True
)
def hit_sphere(center: Point, radius: Real, origin: Point, direction: Vector) -> Real:
    oc = origin - center
    a = length_squared(direction)
    b = dot(oc, direction)
    c = length_squared(oc) - radius * radius
    discriminant = b * b - a * c
    if discriminant < 0.0:
        return Real(-1.0)
    else:
        return Real((-b - np.sqrt(discriminant)) / a)


@nb.njit(real[:](real[:], real[:]), inline="always", parallel=True, cache=True)
def ray_color(origin: Point, direction: Vector) -> np.ndarray:
    t = hit_sphere(point(0, 0, -1), 0.5, origin, direction)

    if t > 0.0:
        n = unit_vector(vector_at_t(origin, direction, t) - vector(0, 0, -1))
        return 0.5 * color(n[0] + 1, n[1] + 1, n[2] + 1)

    unit_direction = unit_vector(direction)
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * WHITE + t * BLUE
