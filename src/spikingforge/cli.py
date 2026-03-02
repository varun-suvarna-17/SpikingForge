import click
from .config import load_config


@click.group()
def cli():
    """SpikingForge CLI"""
    pass


@cli.command()
@click.argument("config_path")
def train(config_path: str):
    """Train SNN using YAML config"""
    click.echo("CLI is working ✅")
    click.echo(f"Config path received: {config_path}")

    config = load_config(config_path)
    click.echo("Loaded config:")
    click.echo(config)


if __name__ == "__main__":
    cli()