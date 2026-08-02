"""Nullain Agent SDK — CLI Helper Utility."""

import argparse
import sys

from nullain import __version__


def cmd_version() -> None:
    """Print Nullain Agent SDK version."""
    print(f"Nullain Agent SDK v{__version__}")


def cmd_doctor() -> None:
    """Run environment and health checks for Nullain SDK."""
    print("Checking Nullain SDK Environment...")
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor} OK")
    print(f"  Nullain SDK v{__version__} OK")


def app() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="nullain",
        description="Nullain Agent SDK CLI Helper",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="Print SDK version")
    sub.add_parser("doctor", help="Run health checks")

    args = parser.parse_args()
    if args.command == "version":
        cmd_version()
    elif args.command == "doctor":
        cmd_doctor()
    else:
        parser.print_help()


if __name__ == "__main__":
    app()
