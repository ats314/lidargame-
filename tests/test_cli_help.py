"""Every --help screen must render.

argparse %-expands help text, so one stray percent sign in one option's help
takes down the entire help screen for that subcommand -- and nothing else. The
compiler still ran, the tests still passed, and `lidarworld compile --help`
raised TypeError: %o format: an integer is required, not dict.
"""
import pytest

from lidarworld.cli import build_parser

SUBCOMMANDS = ["compile", "generate", "validate", "inspect", "themes", "explain", "roles",
               "sources", "fetch", "tiles", "adapters"]


def _subparsers(parser):
    import argparse
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def test_every_subcommand_is_covered():
    """If a subcommand is added, this list has to grow with it."""
    assert set(_subparsers(build_parser())) == set(SUBCOMMANDS)


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_help_renders(name):
    text = _subparsers(build_parser())[name].format_help()
    assert text.strip()
    assert "usage:" in text


def test_top_level_help_renders():
    assert "usage:" in build_parser().format_help()


def test_a_literal_percent_survives_expansion():
    """The specific line that broke, asserted on the rendered output."""
    text = _subparsers(build_parser())["compile"].format_help()
    assert "6% of a downtown grid" in " ".join(text.split())
