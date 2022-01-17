import numpy as np
import numba as nb
from numba.experimental import jitclass
from .vectors import vector, ray_color, real


@nb.njit(parallel=True, cache=True)
def example_scene(height=256, width=256):
    array = np.empty((width, height, 3), dtype=np.float64)

    aspect_ratio = width / height
    viewport_height = 2.0
    viewport_width = aspect_ratio * viewport_height
    focal_length = 1.0

    origin = vector(0, 0, 0)
    horizontal = vector(viewport_width, 0, 0)
    vertical = vector(0, viewport_height, 0)
    lower_left_corner = (
        origin - horizontal / 2 - vertical / 2 - vector(0, 0, focal_length)
    )

    for i in nb.prange(width):
        for j in nb.prange(height):
            u = i / (width - 1)
            v = j / (height - 1)
            destination = lower_left_corner + u * horizontal + v * vertical
            array[i, j, :] = ray_color(origin, destination - origin)

    return array


class Scene(object):
    def __init__(self, array: np.ndarray):
        self.array = array

    @classmethod
    def empty(cls, height=256, width=256):
        array = np.empty((width, height, 3), dtype=np.float64)
        return cls(array)

    @classmethod
    def default(cls, height=256, width=256):
        scene = cls.empty(height=height, width=width)
        array = scene.array

        for j in range(height):
            for i in range(width):
                r = i / (width - 1)
                g = j / (height - 1)
                b = 0.25

                array[i, j, 0] = r
                array[i, j, 1] = g
                array[i, j, 2] = b

        return scene
