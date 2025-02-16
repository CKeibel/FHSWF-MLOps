import pytest


@pytest.mark.parametrize("a", [1])
def test_a(a: int):
    assert 1 == a
