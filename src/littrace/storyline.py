"""Compatibility alias for :mod:`littrace.evidence.storyline`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("littrace.evidence.storyline")
