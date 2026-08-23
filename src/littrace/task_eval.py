"""Compatibility alias for :mod:`littrace.evaluation.task_eval`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("littrace.evaluation.task_eval")
