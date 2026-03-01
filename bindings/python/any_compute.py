"""Auto-generated Python bindings for any_compute_ffi."""
import ctypes
import os
from pathlib import Path

# Load the shared library
_lib_dir = Path(__file__).parent
_lib_name = {
    'linux': 'libany_compute_ffi.so',
    'darwin': 'libany_compute_ffi.dylib',
    'win32': 'any_compute_ffi.dll',
}[__import__('sys').platform]
_lib = ctypes.CDLL(str(_lib_dir / _lib_name))

# Create a new empty VecSource.
_lib.anc_source_new.argtypes = []
_lib.anc_source_new.restype = ctypes.c_void_p

# Add a column definition to a VecSource.
_lib.anc_source_add_column.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint8]
_lib.anc_source_add_column.restype = None

# Push a row of integer values.
_lib.anc_source_push_row_ints.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
_lib.anc_source_push_row_ints.restype = None

# Free a VecSource previously created by anc_source_new.
_lib.anc_source_free.argtypes = [ctypes.c_void_p]
_lib.anc_source_free.restype = None

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
