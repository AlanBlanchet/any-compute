"""Auto-generated tests for {{LIB_NAME}}."""
import pytest
from any_compute import VecSource

def test_create_and_free():
    src = VecSource()
    assert src._handle is not None
    del src  # Should not crash

def test_add_column():
    src = VecSource()
    src.add_column('age', 1)  # Int column
    src.add_column('score', 2)  # Float column

def test_push_rows():
    src = VecSource()
    src.add_column('x', 1)
    src.add_column('y', 1)
    src.push_row_ints([10, 20])
    src.push_row_ints([30, 40])

def test_lifecycle():
    """Full lifecycle: create, populate, destroy."""
    src = VecSource()
    for i in range(100):
        src.add_column(f'col_{i}', i % 3)
    for r in range(1000):
        src.push_row_ints(list(range(100)))
