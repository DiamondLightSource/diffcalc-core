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

from math import atan, radians, sqrt

import pytest
from diffcalc.ub.crystal import Crystal
from numpy import array

from tests.diffcalc import scenarios
from tests.tools import assert_iterable_almost_equal, mneq_


class TestCrystalUnderTest:
    def setup_method(self):
        self.tclatt = []
        self.tcbmat = []

        # From the dif_init.mat next to dif_dos.exe on Vlieg's cd
        # self.tclatt.append([4.0004, 4.0004, 2.270000, 90, 90, 90])
        # self.tcbmat.append([[1.570639, 0, 0] ,[0.0, 1.570639, 0] ,
        #                    [0.0, 0.0, 2.767923]])

        # From b16 on 27June2008 (From Chris Nicklin)
        # self.tclatt.append([3.8401, 3.8401, 5.43072, 90, 90, 90])
        # self.tcbmat.append([[1.636204, 0, 0],[0, 1.636204, 0],
        #                     [0, 0, 1.156971]])

    def testGetBMatrix(self):
        # Check the calculated B Matrix
        for sess in scenarios.sessions():
            if sess.bmatrix is None:
                continue
            cut = Crystal("tc", *sess.lattice)
            desired = array(sess.bmatrix)
            print(desired.tolist())
            answer = cut.B
            print(answer.tolist())
            note = "Incorrect B matrix calculation for scenario " + sess.name
            mneq_(answer, desired, 4, note=note)

    @pytest.mark.parametrize(
        ("xtal_system", "unit_cell", "full_unit_cell"),
        [
            ("Triclinic", (1, 2, 3, 10, 20, 30), (1, 2, 3, 10, 20, 30)),
            ("Monoclinic", (1, 2, 3, 20), (1, 2, 3, 90, 20, 90)),
            ("Orthorhombic", (1, 2, 3), (1, 2, 3, 90, 90, 90)),
            ("Tetragonal", (1, 3), (1, 1, 3, 90, 90, 90)),
            ("Rhombohedral", (1, 10), (1, 1, 1, 10, 10, 10)),
            ("Cubic", (1,), (1, 1, 1, 90, 90, 90)),
            pytest.param(
                "Orthorombic",
                (1, 2, 3),
                (1, 2, 3, 90, 90, 90),
                marks=pytest.mark.xfail(raises=TypeError),
            ),
        ],
    )
    def test_get_lattice_params(self, xtal_system, unit_cell, full_unit_cell):
        xtal = Crystal("xtal", xtal_system, *unit_cell)
        test_xtal_system, test_unit_cell = xtal.get_lattice_params()
        assert test_xtal_system == xtal_system
        assert_iterable_almost_equal(test_unit_cell, unit_cell)

        xtal_name, a1, a2, a3, alpha1, alpha2, alpha3 = xtal.get_lattice()
        assert xtal_name == "xtal"
        assert_iterable_almost_equal(
            (a1, a2, a3, alpha1, alpha2, alpha3), full_unit_cell
        )

    def test_get_hkl_plane_angle(self):
        xtal = Crystal("cube", 1, 1, 1, 90, 90, 90)
        assert xtal.get_hkl_plane_angle((0, 0, 1), (0, 0, 2)) == pytest.approx(0)
        assert xtal.get_hkl_plane_angle((0, 1, 0), (0, 0, 2)) == pytest.approx(
            radians(90)
        )
        assert xtal.get_hkl_plane_angle((1, 0, 0), (0, 0, 2)) == pytest.approx(
            radians(90)
        )
        assert xtal.get_hkl_plane_angle((1, 1, 0), (0, 0, 2)) == pytest.approx(
            radians(90)
        )
        assert xtal.get_hkl_plane_angle((0, 1, 1), (0, 0, 2)) == pytest.approx(
            radians(45)
        )
        assert xtal.get_hkl_plane_angle((1, 0, 1), (0, 0, 2)) == pytest.approx(
            radians(45)
        )
        assert xtal.get_hkl_plane_angle((1, 1, 1), (0, 0, 2)) == pytest.approx(
            atan(sqrt(2.0))
        )

    def test__str__(self):
        cut = Crystal("HCl", 1, 2, 3, 4, 5, 6)
        print(cut.__str__())

    def test_serialisation(self):
        for sess in scenarios.sessions():
            if sess.bmatrix is None:
                continue
            crystal = Crystal("tc", *sess.lattice)
            cut_json = crystal.asdict
            reformed_crystal = Crystal(**cut_json)

            assert (reformed_crystal.B == crystal.B).all()
