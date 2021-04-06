###
# Copyright 2008-2011 Diamond Light Source Ltd.
# This file is part of Diffcalc.
#
# Diffcalc is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Diffcalc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Diffcalc.  If not, see <http://www.gnu.org/licenses/>.
###

import pytest
from diffcalc.ub.calc import ReferenceVector
from diffcalc.util import DiffcalcException
from numpy import array

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
