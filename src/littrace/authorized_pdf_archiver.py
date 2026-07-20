"""Compatibility alias for the access-layer authorized PDF archiver."""

from importlib import import_module
import sys

_implementation = import_module("littrace.access_layer.authorized_pdf_archiver")
sys.modules[__name__] = _implementation
setattr(sys.modules[__package__], "authorized_pdf_archiver", _implementation)
