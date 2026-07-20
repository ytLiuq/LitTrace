"""Compatibility alias for the access-layer CDP implementation."""

from importlib import import_module
import sys

_implementation = import_module("littrace.access_layer.cdp_core")
sys.modules[__name__] = _implementation
setattr(sys.modules[__package__], "cdp_core", _implementation)
