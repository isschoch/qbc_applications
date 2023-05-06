
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qbc_applications.addition import *

import pytest

@pytest.mark.parametrize("a, b", [(x, y) for x in range(2**n_wires) for y in range(2**n_wires)])
def test_addition(a, b):
    assert addition_int_result(a, b) == ((a + b) % 2**n_wires)

@pytest.mark.parametrize("a, b", [(x, y) for x in range(2**n_wires) for y in range(n_wires)])
def test_subtraction(a, b):
    assert subtraction_int_result(a, b) == ((a - b + 2**n_wires) % 2**n_wires)

