"""Compatibility alias for the retrieval full-text implementation."""

from importlib import import_module
import sys

_implementation = import_module("littrace.retrieval.full_text")
sys.modules[__name__] = _implementation
setattr(sys.modules[__package__], "full_text", _implementation)
