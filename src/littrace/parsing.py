"""Compatibility alias for :mod:`littrace.evidence.parsing`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("littrace.evidence.parsing")
