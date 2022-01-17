import click
from .images import Image
from .scenes import example_scene
from .vectors import ray_color, vector
import numpy as np


@click.group()
@click.option("-W", "--width", default=256, type=int, help="Image width.")
@click.option("-H", "--height", default=256, type=int, help="Image height.")
def cli(width, height):

    array = example_scene(width=width, height=height)
    image = Image.from_scene_array(array)

    image.save("test.png")


@click.group()
def cache():
    pass


@cache.command()
def clear():
    import os
    import shutil
    dirname = os.path.dirname(__file__)
    shutil.rmtree(os.path.join(dirname, "__pycache__"))


cli.add_command(cache)

if __name__ == "__main__":
    cli()
