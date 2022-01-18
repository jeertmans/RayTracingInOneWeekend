from PIL import Image as PILImage
import numpy as np


class Image(PILImage.Image):
    @classmethod
    def from_scene_array(cls, scene_array):
        return PILImage.fromarray(np.uint8(np.clip(scene_array, 0., 1.) * 255.999))
