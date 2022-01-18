import numpy as np
import numba as nb
from numba.experimental import jitclass
from typing import Tuple


real = nb.float64
hit = nb.types.Tuple([real[:], real[:], real, nb.boolean])
world = nb.types.Tuple([real[:, :], real[:]])


Real = np.float64
OneDArray = np.ndarray
TwoDArray = np.ndarray
Vector = OneDArray
Color = OneDArray
Point = OneDArray
Hit = Tuple[Point, Vector, Real, bool]
World = Tuple[TwoDArray, OneDArray]

inf = Real("inf")


@nb.njit(real[:](real, real, real), inline="always", cache=True)
def vector(x: Real, y: Real, z: Real) -> Vector:
    return np.array([x, y, z], dtype=Real)


color = vector
point = vector

WHITE = color(1, 1, 1)
BLUE = color(0.5, 0.7, 1)

NO_HIT = point(0, 0, 0), vector(0, 0, 0), Real(0), False


@nb.njit(hit(), inline="always", cache=True)
def no_hit():
    return point(0, 0, 0), vector(0, 0, 0), Real(0), False


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


@nb.njit(real[:](real, real), inline="always", parallel=True, cache=True)
def random_vector(min: Real, max: Real):
    return min + (max - min) * np.random.rand(3).astype(Real)


@nb.njit(real[:](), inline="always", parallel=True, cache=True)
def random_vector_in_unit_sphere():
    while True:
        v = random_vector(-1, 1)
        if length(v) < 1:
            return v


@nb.njit(
    hit(real[:], real, real[:], real[:], real, real),
    inline="always",
    parallel=True,
    cache=True,
)
def hit_sphere(
    center: Point,
    radius: Real,
    origin: Point,
    direction: Vector,
    t_min: Real,
    t_max: Real,
) -> Hit:
    oc = origin - center
    a = length_squared(direction)
    b = dot(oc, direction)
    c = length_squared(oc) - radius * radius
    discriminant = b * b - a * c
    if discriminant < 0.0:
        return no_hit()

    sqrtd = np.sqrt(discriminant)
    root = (-b - sqrtd) / a

    if root < t_min or root > t_max:
        root = (-b + sqrtd) / a
        if root < t_min or root > t_max:
            return no_hit()

    t = root
    p = vector_at_t(origin, direction, t)
    normal = (p - center) / radius
    front_face = dot(direction, normal) < 0
    normal = normal if front_face else -normal

    return p, normal, t, True


@nb.njit(
    hit(world, real[:], real[:], real, real), inline="always", parallel=True, cache=True
)
def hit_world(
    world: World, origin: Point, direction: Vector, t_min: Real, t_max: Real
) -> Hit:

    centers, radiuses = world
    n = centers.shape[0]
    hit_record = no_hit()
    t_closest = t_max

    for i in range(n):
        hit_result = hit_sphere(
            centers[i, :], radiuses[i], origin, direction, t_min, t_closest
        )

        if hit_result[-1]:
            t_closest = hit_result[2]
            hit_record = hit_result

    return hit_record


@nb.njit(
    real[:](real[:], real[:], world, nb.int64),
    inline="never",
    parallel=True,
    cache=True,
)
def ray_color(origin: Point, direction: Vector, world: World, depth: int) -> Color:
    hit_record = hit_world(world, origin, direction, 0, inf)

    if depth <= 0:
        return color(0, 0, 0)

    if hit_record[-1]:
        normal = hit_record[1]
        target = origin + normal + random_vector_in_unit_sphere()
        return 0.5 * ray_color(origin, target - origin, world, depth - 1)

    unit_direction = unit_vector(direction)
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * WHITE + t * BLUE
