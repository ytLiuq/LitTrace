"""Compatibility alias for access-layer path and download planning helpers."""

from importlib import import_module
import sys

_implementation = import_module("littrace.access_layer.paths")
sys.modules[__name__] = _implementation
setattr(sys.modules[__package__], "access", _implementation)
