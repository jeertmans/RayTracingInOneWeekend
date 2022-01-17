import click


@click.group(invoke_without_command=True)
@click.option("-W", "--width", default=256, type=int, help="Image width.")
@click.option("-H", "--height", default=256, type=int, help="Image height.")
@click.option("-S", "--size", default=None, type=int, help="Image size.")
@click.option("-o", "--output", default="out.png", type=click.Path(dir_okay=False), help="Output image file.")
@click.pass_context
def cli(ctx, width, height, size, output):
    if ctx.invoked_subcommand is not None:
        return

    if size is not None:
        width = height = size

    from .images import Image
    from .scenes import example_scene
    from .vectors import ray_color, vector
    import numpy as np

    array = example_scene(width=width, height=height)
    image = Image.from_scene_array(array)

    image.save(output)


@cli.command()
def clear_cache():
    import os
    import shutil

    dirname = os.path.dirname(__file__)
    shutil.rmtree(os.path.join(dirname, "__pycache__"))
    click.secho("Cache folder was cleared!", fg="green")


if __name__ == "__main__":
    cli()
