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

from numpy import array

from tests.tools import (
    assert_2darray_almost_equal,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_dict_almost_equal,
    assert_matrix_almost_equal,
)


def eq_(a, b):
    assert a == b, f"{a!r} != {b!r}"


class test_assert_almost_equal:
    def test_fails(self):
        try:
            assert_almost_equal(1, 1.0000001)
            assert False
        except AssertionError:
            #            eq_(e.args[0], '1 != 1.0000001000000001 within 7 places')
            pass

    def test__passes(self):
        assert_almost_equal(1, 1.00000001)
        assert_almost_equal(1, 1.0000001, 6)


class test_assert_array_almost_equal:
    def test__passes(self):
        assert_array_almost_equal((1, 2, 3), (1.00000001, 2, 3.0))
        assert_array_almost_equal((1, 2, 3), (1.0000001, 2, 3.0), 6)
        assert_array_almost_equal((1, 2, 3), [1, 2, 3])
        assert_array_almost_equal((), ())
        assert_array_almost_equal([], ())

    def test_wrong_length(self):
        try:
            assert_array_almost_equal((1, 2, 3), (1, 2))
            assert False
        except AssertionError as e:
            eq_(e.args[0], "(1, 2, 3) != (1, 2) as lengths differ")

    def test_wrong_value(self):
        try:
            assert_array_almost_equal((3, 2, 1), (3, 2, 1.0000001))
            assert False
        except AssertionError:
            pass


class test_assert_2darray_almost_equal:
    def test__passes(self):
        assert_2darray_almost_equal(((1, 2), (3, 4)), ((1.00000001, 2), (3, 4)))
        assert_2darray_almost_equal(((1, 2), (3, 4)), ((1.0000001, 2), (3, 4)), 6)
        assert_2darray_almost_equal(((1, 2), (3, 4)), [(1, 2), [3, 4]])
        assert_2darray_almost_equal(((), ()), ((), ()))
        assert_2darray_almost_equal([(), []], ([], ()))

    def test_wrong_length(self):
        try:
            assert_2darray_almost_equal(((1, 2), (3, 4)), ((1, 2), (3, 4, 5)))
            assert False
        except AssertionError as e:
            eq_(e.args[0], "((1, 2), (3, 4)) != ((1, 2), (3, 4, 5)) as sizes differ")

    def test_wrong_value(self):
        try:
            assert_2darray_almost_equal(((1, 2), (3, 4)), ((1.0000001, 2), (3, 4)))
            assert False
        except AssertionError:
            pass


class test_assert_matrix_almost_equal:
    def test__passes(self):
        assert_matrix_almost_equal(
            array([[1, 2], [3, 4]]), array([[1.00000001, 2], [3, 4]])
        )
        assert_matrix_almost_equal(
            array([[1, 2], [3, 4]]), array([[1.0000001, 2], [3, 4]]), 6
        )

    def test_wrong_length(self):
        try:
            assert_matrix_almost_equal(
                array([[1, 2], [3, 4]]), array([[1, 2, 3], [4, 5, 6]])
            )
            assert False
        except AssertionError:
            pass

    def test_wrong_value(self):
        try:
            assert_matrix_almost_equal(
                array([[1, 2], [3, 4]]), array([[1.0000001, 2], [3, 4]])
            )
            assert False
        except AssertionError:
            pass


class test_assert_dict_almost_equal:
    def test__passes(self):
        assert_dict_almost_equal({}, {})
        assert_dict_almost_equal({"a": 1}, {"a": 1})
        assert_dict_almost_equal({"a": 1.0}, {"a": 1.0})
        assert_dict_almost_equal({"a": 1.00000001}, {"a": 1.00000001})

    def test_wrong_keys(self):
        try:
            assert_dict_almost_equal({"a": 1}, {"a": 1, "b": 2})
            assert False
        except AssertionError:
            pass

    def test_wrong_object(self):
        try:
            assert_dict_almost_equal({"a": 1}, {"a": "1"})
            assert False
        except AssertionError as e:
            eq_(e.args[0], "For key 'a', 1 != '1'")

    def test_wrong_int(self):
        try:
            assert_dict_almost_equal({"a": 1}, {"a": 2})
            assert False
        except AssertionError as e:
            eq_(e.args[0], "For key 'a', 1 != 2")

    def test_wrong_float1(self):
        try:
            assert_dict_almost_equal({"a": 1.0}, {"a": 2.0})
            assert False
        except AssertionError as e:
            eq_(e.args[0], "For key 'a', 1.0 != 2.0 within 7 places")

    def test_wrong_float2(self):
        try:
            assert_dict_almost_equal({"a": 1}, {"a": 2.0})
            assert False
        except AssertionError as e:
            eq_(e.args[0], "For key 'a', 1 != 2.0 within 7 places")

    def test_wrong_float3(self):
        try:
            assert_dict_almost_equal({"a": 1.0}, {"a": 2})
            assert False
        except AssertionError as e:
            eq_(e.args[0], "For key 'a', 1.0 != 2 within 7 places")

    def test_okay_differing_types1(self):
        assert_dict_almost_equal({"a": 1.0}, {"a": 1})

    def test_okay_differing_types2(self):
        assert_dict_almost_equal({"a": 1}, {"a": 1.0})
