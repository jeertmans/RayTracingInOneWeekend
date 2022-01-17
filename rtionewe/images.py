from PIL import Image as PILImage
import numpy as np


class Image(PILImage.Image):
    @classmethod
    def from_scene_array(cls, scene_array):
        return PILImage.fromarray(np.uint8(scene_array * 255.999))
