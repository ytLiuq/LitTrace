"""Compatibility alias for the access-layer login implementation."""

from importlib import import_module
import sys

_implementation = import_module("littrace.access_layer.login_flow")
sys.modules[__name__] = _implementation
setattr(sys.modules[__package__], "login_flow", _implementation)
