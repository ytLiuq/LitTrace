"""Compatibility alias for :mod:`littrace.evaluation.pdf_benchmark`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("littrace.evaluation.pdf_benchmark")
