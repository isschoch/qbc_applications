
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from addition import *
from qbc import *
import math
import pytest

@pytest.mark.parametrize("a, b", [(x, y) for x in range(2**n_wires) for y in range(2**n_wires)])
def test_addition(a, b):
    assert addition_int_result(a, b) == ((a + b) % 2**n_wires)

@pytest.mark.parametrize("a, b", [(x, y) for x in range(2**n_wires) for y in range(n_wires)])
def test_subtraction(a, b):
    assert subtraction_int_result(a, b) == ((a - b + 2**n_wires) % 2**n_wires)

@pytest.mark.parametrize("n, num_t_wires", [(n, num_t_wires) for n in range(2, 6) for num_t_wires in range(10, 11)])
def test_qbc_algorithm(n, num_t_wires):
    x = np.random.randint(0, 2, size=(2**n))
    y = np.random.randint(0, 2, size=(2**n))
    rho, _, _, _, _ = qbc_algorithm(x, y, num_t_wires)
    assert math.isclose(rho, np.inner(x, y), rel_tol=1e-3)

def test_qbc_algorithm():
    x = [1, 0, 0, 1, 0, 1, 1, 1, 0]
    y = [0, 0, 1, 1, 1, 1, 1, 0, 1]
    rho, _, _, _, _ = qbc_algorithm(x, y)
    assert math.isclose(rho, np.inner(x, y), rel_tol=1e-3)
