import pytest
from diffcalc.ub.calc import ReferenceVector
from diffcalc.util import DiffcalcException
from numpy import array
from numpy.linalg import inv

from tests.tools import assert_2darray_almost_equal


@pytest.fixture
def reference():
    return ReferenceVector((0, 0, 1), False)


@pytest.mark.parametrize(
    ("vector"),
    [
        array([[1], [2], [3]]),
        pytest.param([[1], [2]], marks=pytest.mark.xfail(raises=DiffcalcException)),
        pytest.param(
            array([[1], [2]]), marks=pytest.mark.xfail(raises=DiffcalcException)
        ),
    ],
)
def test_from_as_array(reference, vector):
    reference.set_array(vector)
    assert reference.n_ref == tuple(vector.T[0])
    result = reference.get_array()
    assert_2darray_almost_equal(vector, result)


@pytest.mark.parametrize(
    ("vector", "UB"),
    [
        (array([[1], [0], [0]]), array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])),
        pytest.param(
            array([[1], [0], [0]]),
            array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]),
            marks=pytest.mark.xfail(raises=DiffcalcException),
        ),
        pytest.param(
            array([[1], [0], [0]]),
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
            marks=pytest.mark.xfail(raises=DiffcalcException),
        ),
    ],
)
def test_from_as_array_UB(vector, UB):
    for rlv in (False, True):
        reference = ReferenceVector((0, 0, 1), rlv)
        reference.set_array(vector)
        result = reference.get_array(UB)
        if rlv:
            assert_2darray_almost_equal(inv(UB) @ vector, result)
        else:
            assert_2darray_almost_equal(UB @ vector, result)


def test_default_n_phi(reference):
    assert_2darray_almost_equal(reference.get_array(), array([[0], [0], [1]]).tolist())


def test__str__with_phi_configured(reference):
    print(reference)


def test__str__with_hkl_configured(reference):
    reference = ReferenceVector((0, 1, 1), True)
    print(reference)


def test_n_phi_from_hkl_with_unity_matrix_001(reference):
    reference = ReferenceVector((0, 0, 1), True)
    assert_2darray_almost_equal(reference.get_array(), array([[0], [0], [1]]))


def test_n_phi_from_hkl_with_unity_matrix_010(reference):
    reference = ReferenceVector((0, 1, 0), True)
    assert_2darray_almost_equal(reference.get_array(), array([[0], [1], [0]]))


def test_serialisation(reference):
    reference_json = reference.asdict
    reformed_reference = ReferenceVector(**reference_json)

    assert reformed_reference.asdict == reference.asdict
