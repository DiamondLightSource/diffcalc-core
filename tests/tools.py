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

from diffcalc.util import SMALL, TORAD, radians_equivalent
from pytest import approx


#
#
def format_note(note):
    return " # %s" % note if note else ""


def assert_almost_equal(first, second, places=7, msg=None):
    if msg:
        assert first == approx(second, rel=pow(10, -places), abs=pow(10, -places)), msg
    else:
        assert first == approx(second, rel=pow(10, -places), abs=pow(10, -places))


def assert_array_almost_equal(first, second, places=7, msg=None, note=None):
    err = "{!r} != {!r} as lengths differ{}".format(first, second, format_note(note))
    assert len(first) == len(second), err
    for f, s in zip(first, second):
        # default_msg = "\n%r != \n%r within %i places%s" % (
        #    format_array(first, places),
        #    format_array(second, places),
        #    places,
        #    format_note(note),
        # )
        # assert_almost_equal(f, s, places, msg or default_msg)
        assert radians_equivalent(f, s, pow(10, -places))


def assert_array_almost_equal_in_list(expected, vals, places=7, msg=None):
    for val in vals:
        try:
            assert_array_almost_equal(val, expected, places)
            return
        except AssertionError:
            continue
    if msg:
        raise AssertionError(msg)
    else:
        raise AssertionError(
            f"{expected} not equal to any element in {vals} within {places} places"
        )


def format_array(a, places):
    fmt = "% ." + str(places) + "f"
    return "(" + ", ".join((fmt % el) for el in a) + ")"


aneq_ = arrayeq_ = assert_array_almost_equal


def assert_2darray_almost_equal(first, second, places=7, msg=None, note=None):
    err = "{!r} != {!r} as sizes differ{}".format(first, second, format_note(note))
    assert len(first) == len(second), err
    for f2, s2 in zip(first, second):
        err = "{!r} != {!r} as sizes differ{}".format(first, second, format_note(note))
        assert len(f2) == len(s2), err
        for f, s in zip(f2, s2):
            message = "within %i places%s" % (places, format_note(note))
            message += "\n" + format_2darray(first, places) + "!=\n"
            message += format_2darray(second, places)
            assert_almost_equal(f, s, places, msg or message)


def format_2darray(array, places):
    fmt = "% ." + str(places) + "f"
    s = ""
    for row in array:
        line = [fmt % el for el in row]
        s += "[" + ", ".join(line) + "]\n"
    return s


def assert_matrix_almost_equal(first, second, places=7, msg=None, note=None):
    assert_2darray_almost_equal(first, second, places, msg)


def assert_dict_almost_equal(first, second, places=7, msg=None, note=None):
    def_msg = "{!r} != {!r} as keys differ{}".format(first, second, format_note(note))
    assert set(first.keys()) == set(second.keys()), msg or def_msg
    keys = list(first.keys())
    keys.sort()
    for key in keys:
        f = first[key]
        s = second[key]
        if isinstance(f, float) or isinstance(s, float):
            def_msg = "For key %s, %r != %r within %i places%s" % (
                repr(key),
                f,
                s,
                places,
                format_note(note),
            )
            assert_almost_equal(f, s, places, msg or def_msg)
        else:
            if f != s:
                raise AssertionError(
                    "For key {}, {!r} != {!r}{}".format(
                        repr(key), f, s, format_note(note)
                    )
                )


dneq_ = assert_dict_almost_equal


def assert_second_dict_almost_in_first(value, expected, places=7, msg=None):
    value = {k: v for k, v in value.items() if k in expected}
    assert_dict_almost_equal(value, expected, places=places, msg=msg)


def assert_dict_almost_in_list(vals, expected, places=7, msg=None):
    for val in vals:
        try:
            assert_second_dict_almost_in_first(val, expected, places, msg)
            return
        except AssertionError:
            continue
    if msg:
        raise AssertionError(msg)
    else:
        raise AssertionError(
            f"{expected} not in any element in {vals} within {places} places"
        )


def assert_iterable_almost_equal(first, second, places=7, msg=None, note=None):
    def_msg = "{!r} != {!r} as lengths differ{}".format(
        first, second, format_note(note)
    )
    assert len(first) == len(second), msg or def_msg
    for f, s in zip(first, second):
        if isinstance(f, float) or isinstance(s, float):
            assert_almost_equal(f, s, places, msg)
        else:
            if f != s:
                raise AssertionError("{!r} != {!r}{}".format(f, s, format_note(note)))


mneq_ = matrixeq_ = assert_matrix_almost_equal


def degrees_equivalent(first, second, tolerance=SMALL):
    return radians_equivalent(first * TORAD, second * TORAD, tolerance * TORAD)
