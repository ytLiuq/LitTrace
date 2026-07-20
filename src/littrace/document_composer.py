"""Compatibility alias for :mod:`littrace.evidence.document_composer`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("littrace.evidence.document_composer")
