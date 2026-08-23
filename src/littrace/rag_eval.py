"""Compatibility alias for :mod:`littrace.evaluation.rag_eval`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("littrace.evaluation.rag_eval")
