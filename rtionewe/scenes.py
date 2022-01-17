import numpy as np


class Scene(object):
    def __init__(self, array: np.ndarray):
        self.array = array

    @classmethod
    def default(cls, height=256, width=256):
        array = np.empty((width, height, 3), dtype=np.float64)

        for j in range(height):
            for i in range(width):
                r = i / (width - 1)
                g = j / (height - 1)
                b = 0.25

                array[i, j, 0] = r
                array[i, j, 1] = g
                array[i, j, 2] = b

        return cls(array)
