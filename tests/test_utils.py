import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from addition import *
from addition_two_wires import *
from qbc import *
from qbc_extended_space import *
from qbc_hamming_dist import *
from qbc_conv import *
from scipy.spatial.distance import hamming
import math
import pytest


@pytest.mark.parametrize(
    "a, b", [(x, y) for x in range(2**num_n_wires) for y in range(2**num_n_wires)]
)
def test_addition(a, b):
    assert addition_int_result(a, b) == ((a + b) % 2**num_n_wires)


@pytest.mark.parametrize(
    "a, b", [(x, y) for x in range(2**num_n_wires) for y in range(2**num_n_wires)]
)
def test_subtraction(a, b):
    assert subtraction_int_result(a, b) == (
        (b - a + 2**num_n_wires) % 2**num_n_wires
    )


# @pytest.mark.parametrize("a, b", [(x, y) for x in range(2**num_n_wires) for y in range(2**num_n_wires)])
# def test_addition_two_wires(a, b):
#     assert addition_two_wires_int_result(a, b) == ((a + b) % 2**num_n_wires)


@pytest.mark.parametrize(
    "a, b", [(x, y) for x in range(2**num_n_wires) for y in range(num_n_wires)]
)
def test_subtraction_two_wires(a, b):
    assert subtraction_two_wires_int_result(a, b) == (
        (b - a + 2**num_n_wires) % 2**num_n_wires
    )


@pytest.mark.parametrize(
    "n, num_t_wires",
    [(n, num_t_wires) for n in range(2, 6) for num_t_wires in range(10, 11)],
)
def test_qbc_algorithm(n, num_t_wires):
    x = np.random.randint(0, 2, size=(2**n))
    y = np.random.randint(0, 2, size=(2**n))
    rho, _, _, _, _ = qbc_algorithm(x, y, num_t_wires)
    assert math.isclose(rho, np.inner(x, y), rel_tol=1e-1)


@pytest.mark.parametrize(
    "n, num_t_wires",
    [(n, num_t_wires) for n in range(2, 4) for num_t_wires in range(10, 11)],
)
def test_qbc_algorithm_extended_space(n, num_t_wires):
    x = np.random.randint(0, 2, size=(2**n))
    y = np.random.randint(0, 2, size=(2**n))
    rho, _, _, _, _ = qbc_conv_algorithm_extended_space(x, y, num_t_wires)
    assert math.isclose(rho, np.inner(x, y), rel_tol=1e-1)


@pytest.mark.parametrize(
    "n, num_t_wires",
    [(n, num_t_wires) for n in range(2, 4) for num_t_wires in range(9, 10)],
)
def test_qbc_conv(n, num_t_wires):
    x = np.random.randint(0, 2, size=(2**n))
    y = np.random.randint(0, 2, size=(2**n))
    analytical_result = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)))
    for i in range(len(x)):
        rho, _, _, _, _ = qbc_conv(x, y, i, num_t_wires)
        assert math.isclose(rho, analytical_result[i], rel_tol=1e-1)


@pytest.mark.parametrize(
    "n, num_t_wires",
    [(n, num_t_wires) for n in range(2, 6) for num_t_wires in range(10, 11)],
)
def test_qbc_algorithm_hamming(n, num_t_wires):
    x = np.random.randint(0, 2, size=(2**n))
    y = np.random.randint(0, 2, size=(2**n))
    rho, _, _, _, _ = qbc_algorithm_hamming(x, y, num_t_wires)
    assert math.isclose(rho, hamming(x, y) * len(x), rel_tol=1e-1)


@pytest.mark.parametrize("N", [N for N in range(4, 12)])
def test_qbc_algorithm_weird_lengths(N):
    # x = [1, 0, 0, 1, 0, 1, 1, 1, 0]
    # y = [0, 0, 1, 1, 1, 1, 1, 0, 1]
    x = np.random.randint(0, 2, size=(N))
    y = np.random.randint(0, 2, size=(N))
    rho, _, _, _, _ = qbc_algorithm(x, y, num_t_wires=10)
    assert math.isclose(rho, np.inner(x, y), rel_tol=1e-1)
