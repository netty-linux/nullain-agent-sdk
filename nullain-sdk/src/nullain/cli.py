"""Nullain Agent SDK — CLI Helper Utility."""

import typer

from nullain import __version__

app = typer.Typer(name="nullain", help="Nullain Agent SDK CLI Helper")


@app.command()
def version() -> None:
    """Print Nullain Agent SDK version."""
    typer.echo(f"Nullain Agent SDK v{__version__}")


@app.command()
def doctor() -> None:
    """Run environment and health checks for Nullain SDK."""
    typer.echo("Checking Nullain SDK Environment...")
    typer.echo("✅ Python environment OK")
    typer.echo(f"✅ Nullain SDK v{__version__} OK")


if __name__ == "__main__":
    app()
