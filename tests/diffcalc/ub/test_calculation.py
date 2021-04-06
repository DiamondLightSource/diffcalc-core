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

from math import pi

from diffcalc.ub.calc import UBCalculation
from numpy import array

from tests.diffcalc.scenarios import PosFromI16sEuler
from tests.test_tools import eq_
from tests.tools import matrixeq_

UB1 = (
    array(
        (
            (0.9996954135095477, -0.01745240643728364, -0.017449748351250637),
            (0.01744974835125045, 0.9998476951563913, -0.0003045864904520898),
            (0.017452406437283505, -1.1135499981271473e-16, 0.9998476951563912),
        )
    )
    * (2 * pi)
)

EN1 = 12.39842
REF1a = PosFromI16sEuler(1, 1, 30, 0, 60, 0)
REF1b = PosFromI16sEuler(1, 91, 30, 0, 60, 0)


def testAgainstI16Results():
    ubcalc = UBCalculation("cubcalc")
    ubcalc.set_lattice("latt", 1, 1, 1, 90, 90, 90)
    ubcalc.add_reflection((1, 0, 0), REF1a, EN1, "100")
    ubcalc.add_reflection((0, 0, 1), REF1b, EN1, "001")
    ubcalc.calc_ub()
    matrixeq_(ubcalc.UB, UB1)


def test_save_and_restore_empty_ubcalc(tmpdir):
    NAME = "test_save_and_restore_empty_ubcalc_with_one_already_started"
    ubcalc = UBCalculation(NAME)

    test_file = tmpdir / "test.pkl"
    ubcalc.pickle(test_file)

    ubcalc2 = UBCalculation.load(test_file)

    eq_(ubcalc2.name, NAME)


def test_save_and_restore_ubcalc_with_lattice(tmpdir):
    NAME = "test_save_and_restore_ubcalc_with_lattice"
    ubcalc = UBCalculation(NAME)
    ubcalc.set_lattice("latt", 1, 1, 1, 90, 90, 90)

    test_file = tmpdir / "test.pkl"
    ubcalc.pickle(test_file)

    ubcalc2 = UBCalculation.load(test_file)

    eq_(ubcalc2.crystal.get_lattice(), ("latt", 1, 1, 1, 90, 90, 90))


def test_save_and_restore_ubcalc_with_reflections(tmpdir):
    NAME = "test_save_and_restore_ubcalc_with_reflections"
    ubcalc = UBCalculation(NAME)
    ubcalc.add_reflection((1, 0, 0), REF1a, EN1, "100")
    ubcalc.add_reflection((0, 0, 1), REF1b, EN1, "001")
    ubcalc.add_reflection((0, 0, 1.5), REF1b, EN1, "001_5")
    ref1 = ubcalc.get_reflection(1)
    ref2 = ubcalc.get_reflection(2)
    ref3 = ubcalc.get_reflection(3)

    filename = tmpdir / "test_file.pkl"
    ubcalc.pickle(filename)

    ubcalc2 = UBCalculation.load(filename)

    eq_(ubcalc2.get_reflection(1), ref1)
    eq_(ubcalc2.get_reflection(2), ref2)
    eq_(ubcalc2.get_reflection(3), ref3)


def test_save_and_restore_ubcalc_with_UB_from_two_ref(tmpdir):
    NAME = "test_save_and_restore_ubcalc_with_UB_from_two_ref"
    ubcalc = UBCalculation(NAME)
    ubcalc.set_lattice("latt", 1, 1, 1, 90, 90, 90)
    ubcalc.add_reflection((1, 0, 0), REF1a, EN1, "100")
    ubcalc.add_reflection((0, 0, 1), REF1b, EN1, "001")
    ubcalc.calc_ub()
    matrixeq_(ubcalc.UB, UB1)

    filename = tmpdir / "test_file.pkl"
    ubcalc.pickle(filename)
    ubcalc2 = UBCalculation.load(filename)

    matrixeq_(ubcalc2.UB, ubcalc.UB)


def test_save_and_restore_ubcalc_with_UB_from_one_ref(tmpdir):
    NAME = "test_save_and_restore_ubcalc_with_UB_from_one_ref"
    ubcalc = UBCalculation(NAME)
    ubcalc.set_lattice("latt", 1, 1, 1, 90, 90, 90)
    ubcalc.add_reflection((1, 0, 0), REF1a, EN1, "100")
    ubcalc.calc_ub()
    matrixeq_(ubcalc.UB, UB1, places=2)

    filename = tmpdir / "test_file.pkl"
    ubcalc.pickle(filename)

    ubcalc2 = UBCalculation.load(filename)

    matrixeq_(ubcalc2.UB, ubcalc.UB)


def test_save_and_restore_ubcalc_with_manual_ub(tmpdir):
    NAME = "test_save_and_restore_ubcalc_with_manual_ub"
    UB = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ubcalc = UBCalculation(NAME)
    ubcalc.set_ub(UB)
    matrixeq_(ubcalc.UB, UB)

    filename = tmpdir / "test_file.pkl"
    ubcalc.pickle(filename)

    ubcalc2 = UBCalculation.load(filename)

    matrixeq_(ubcalc2.UB, ubcalc.UB)


def test_save_and_restore_ubcalc_with_manual_u(tmpdir):
    NAME = "test_save_and_restore_ubcalc_with_manual_u"
    U = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ubcalc = UBCalculation(NAME)
    ubcalc.set_lattice("latt", 1, 1, 1, 90, 90, 90)
    ubcalc.set_u(U)
    matrixeq_(ubcalc.UB, U * 2 * pi)
    filename = tmpdir / "test_file.pkl"
    ubcalc.pickle(filename)

    ubcalc2 = UBCalculation.load(filename)

    matrixeq_(ubcalc2.UB, ubcalc.UB)
