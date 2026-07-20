"""Compatibility alias for the access-layer browser implementation."""

from importlib import import_module
import sys

_implementation = import_module("littrace.access_layer.browser")
sys.modules[__name__] = _implementation
setattr(sys.modules[__package__], "browser", _implementation)
