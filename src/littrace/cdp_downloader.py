"""Compatibility alias for the access-layer CDP downloader."""

from importlib import import_module
import sys

_implementation = import_module("littrace.access_layer.cdp_downloader")
sys.modules[__name__] = _implementation
setattr(sys.modules[__package__], "cdp_downloader", _implementation)
