"""Compatibility alias for retrieval source routing."""

from importlib import import_module
import sys

_implementation = import_module("littrace.retrieval.source_router")
sys.modules[__name__] = _implementation
setattr(sys.modules[__package__], "source_router", _implementation)
