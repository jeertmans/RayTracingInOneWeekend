import click
from .images import Image
from .scenes import Scene
import numpy as np


@click.command()
@click.option("-W", "--width", default=256, type=int, help="Image width.")
@click.option("-H", "--height", default=256, type=int, help="Image height.")
def cli(width, height):

    array = Scene.default(width, height).array
    image = Image.from_scene_array(array)

    image.save("test.png")


if __name__ == "__main__":
    cli()
