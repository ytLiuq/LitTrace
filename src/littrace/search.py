"""Compatibility alias for the retrieval search implementation."""

from importlib import import_module
import sys

_implementation = import_module("littrace.retrieval.search")
sys.modules[__name__] = _implementation
setattr(sys.modules[__package__], "search", _implementation)
