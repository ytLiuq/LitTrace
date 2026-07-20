"""Compatibility alias for :mod:`littrace.evaluation.golden_eval`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("littrace.evaluation.golden_eval")
