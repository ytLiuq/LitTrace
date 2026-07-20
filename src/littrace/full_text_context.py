"""Compatibility alias for retrieval full-text context construction."""

from importlib import import_module
import sys

_implementation = import_module("littrace.retrieval.full_text_context")
sys.modules[__name__] = _implementation
setattr(sys.modules[__package__], "full_text_context", _implementation)
