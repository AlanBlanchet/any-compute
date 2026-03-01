"""Auto-generated Python bindings for {{LIB_NAME}}."""
import ctypes
import os
from pathlib import Path

# Load the shared library
_lib_dir = Path(__file__).parent
_lib_name = {
    'linux': 'lib{{LIB_NAME}}.so',
    'darwin': 'lib{{LIB_NAME}}.dylib',
    'win32': '{{LIB_NAME}}.dll',
}[__import__('sys').platform]
_lib = ctypes.CDLL(str(_lib_dir / _lib_name))

{{FUNCTION_DECLARATIONS}}

class VecSource:
    """Pythonic wrapper around the C VecSource handle."""

    def __init__(self):
        self._handle = _lib.anc_source_new()

    def __del__(self):
        if self._handle:
            _lib.anc_source_free(self._handle)
            self._handle = None

    def add_column(self, name: str, kind: int = 1):
        _lib.anc_source_add_column(self._handle, name.encode('utf-8'), kind)

    def push_row_ints(self, values: list[int]):
        arr = (ctypes.c_int64 * len(values))(*values)
        _lib.anc_source_push_row_ints(self._handle, arr, len(values))
