"""
Console module.
"""
import argparse
from pathlib import Path

from rich.console import Console

console = Console()


def parse_cli() -> argparse.Namespace:
    """Parse cli to get input yaml conf file.

    Returns:
        argparse.Namespace: a dictionary containing the representation of the yaml file
    """
    parser = argparse.ArgumentParser(
        Path(__file__).stem,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--conf",
        help="Configuration Path",
        type=Path,
        default=Path("params.yaml"),
    )

    return parser.parse_args()
