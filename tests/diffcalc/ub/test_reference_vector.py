import numpy as np
import pytest
from diffcalc.ub.calc import ReferenceVector
from diffcalc.util import DiffcalcException
from numpy.linalg import inv


@pytest.fixture
def reference():
    return ReferenceVector((0, 0, 1), False)


@pytest.mark.parametrize(
    ("vector"),
    [
        pytest.param([[1], [2]]),
        pytest.param(np.array([[1], [2]])),
    ],
)
def test_reference_vector_set_array_fails_for_incorrect_params(reference, vector):
    with pytest.raises(DiffcalcException):
        reference.set_array(vector)


@pytest.mark.parametrize(
    ("ub"),
    [
        pytest.param([[1], [2]]),
        pytest.param(np.array([[1, 2], [4, 5], [7, 8]])),
    ],
)
def test_reference_vector_get_array_fails_for_incorrect_params(reference, ub):
    with pytest.raises(DiffcalcException):
        reference.get_array(ub)


def test_reference_vector_get_array_retrieves_expected_array(reference):
    expected_array = reference.get_array()
    assert np.all(expected_array == np.array([[0], [0], [1]]))


@pytest.mark.parametrize(
    ("vector", "ub"),
    [
        (np.array([[1], [0], [0]]), np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])),
    ],
)
def test_reference_vector_reciprocal_vs_laboratory_frame(vector, ub):
    for rlv in (False, True):
        reference = ReferenceVector((0, 0, 1), rlv)
        reference.set_array(vector)
        result = reference.get_array(ub)
        if rlv:
            assert np.all(inv(ub) @ vector == result)
        else:
            assert np.all(ub @ vector == result)
