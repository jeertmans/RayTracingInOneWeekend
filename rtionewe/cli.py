import click


@click.group(invoke_without_command=True)
@click.option("-W", "--width", default=256, type=int, help="Image width.")
@click.option("-H", "--height", default=256, type=int, help="Image height.")
@click.pass_context
def cli(ctx, width, height):
    if ctx.invoked_subcommand is not None:
        return

    from .images import Image
    from .scenes import example_scene
    from .vectors import ray_color, vector
    import numpy as np

    array = example_scene(width=width, height=height)
    image = Image.from_scene_array(array)

    image.save("test.png")


@cli.command()
def clear_cache():
    import os
    import shutil

    dirname = os.path.dirname(__file__)
    shutil.rmtree(os.path.join(dirname, "__pycache__"))
    click.secho("Cache folder was cleared!", fg="green")


if __name__ == "__main__":
    cli()
