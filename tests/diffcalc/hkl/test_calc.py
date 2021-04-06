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

import dataclasses
import itertools
from itertools import chain
from math import cos, pi, sin

import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import TORAD, DiffcalcException, I, y_rotation, z_rotation
from numpy import array

from tests.diffcalc.scenarios import Pos, PosFromI16sEuler
from tests.tools import (
    arrayeq_,
    assert_array_almost_equal,
    assert_array_almost_equal_in_list,
    assert_dict_almost_in_list,
    assert_second_dict_almost_in_first,
)


class Pair:
    def __init__(self, name, hkl, position, zrot, yrot, wavelength):
        self.name = name
        self.hkl = hkl
        self.position = position
        self.zrot = zrot
        self.yrot = yrot
        self.wavelength = wavelength


def create_ubcalc():
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)
    ubcalc.surf_nphi = (0, 0, 1)
    return ubcalc


def test_str():
    ubcalc = create_ubcalc()
    ubcalc = UBCalculation("test_str")
    ubcalc.n_phi = (0, 0, 1)
    ubcalc.surf_nphi = (0, 0, 1)
    ubcalc.set_lattice("xtal", "Cubic", 1)
    ubcalc.add_reflection((0, 0, 1), Position(0, 60, 0, 30, 0, 0), 12.4, "ref1")
    ubcalc.add_orientation((0, 1, 0), (0, 1, 0), Position(1, 0, 0, 0, 2, 0), "orient1")
    ubcalc.set_u(I)

    constraints = Constraints()
    constraints.nu = 0
    constraints.psi = 90
    constraints.phi = 90

    hklcalc = HklCalculation(ubcalc, constraints)
    assert (
        str(hklcalc)
        == """    DET             REF             SAMP
    -----------     -----------     -----------
    delta           a_eq_b          mu
--> nu              alpha           eta
    qaz             beta            chi
    naz         --> psi         --> phi
                    bin_eq_bout     bisect
                    betain          omega
                    betaout

    nu   : 0.0000
    psi  : 90.0000
    phi  : 90.0000
"""
    )


class _BaseTest:
    def setup_method(self):
        self.ubcalc = create_ubcalc()
        self.constraints = Constraints()
        self.hklcalc = HklCalculation(self.ubcalc, self.constraints)

        self.places = 5

    def _configure_ub(self):
        ZROT = z_rotation(self.zrot * TORAD)  # -PHI
        YROT = y_rotation(self.yrot * TORAD)  # +CHI
        U = ZROT @ YROT
        # UB = U @ self.B
        self.ubcalc.set_u(U)  # self.mock_ubcalc.UB = UB

    def _check_hkl_to_angles(
        self, testname, zrot, yrot, hkl, pos_expected, wavelength, virtual_expected={}
    ):
        print(
            "_check_hkl_to_angles(%s, %.1f, %.1f, %s, %s, %.2f, %s)"
            % (testname, zrot, yrot, hkl, pos_expected, wavelength, virtual_expected)
        )
        self.zrot, self.yrot = zrot, yrot
        self._configure_ub()

        pos_virtual_angles_pairs_in_degrees = self.hklcalc.get_position(
            hkl[0], hkl[1], hkl[2], wavelength
        )
        pos = list(chain(*pos_virtual_angles_pairs_in_degrees))[::2]
        virtual = list(chain(*pos_virtual_angles_pairs_in_degrees))[1::2]
        assert_array_almost_equal_in_list(
            dataclasses.astuple(pos_expected),
            [dataclasses.astuple(p) for p in pos],
            self.places,
        )
        # assert_array_almost_equal(pos, pos_expected, self.places)
        assert_dict_almost_in_list(virtual, virtual_expected)

    def _check_angles_to_hkl(
        self, testname, zrot, yrot, hkl_expected, pos, wavelength, virtual_expected={}
    ):
        print(
            "_check_angles_to_hkl(%s, %.1f, %.1f, %s, %s, %.2f, %s)"
            % (testname, zrot, yrot, hkl_expected, pos, wavelength, virtual_expected)
        )
        self.zrot, self.yrot = zrot, yrot
        self._configure_ub()
        hkl = self.hklcalc.get_hkl(pos, wavelength)
        virtual = self.hklcalc.get_virtual_angles(pos)
        assert_array_almost_equal(
            hkl,
            hkl_expected,
            self.places,
            note="***Test (not diffcalc!) incorrect*** : the desired settings do not map to the target hkl",
        )
        assert_second_dict_almost_in_first(virtual, virtual_expected)

    def case_generator(self, case):
        self._check_angles_to_hkl(
            case.name,
            case.zrot,
            case.yrot,
            case.hkl,
            case.position,
            case.wavelength,
            {},
        )
        self._check_hkl_to_angles(
            case.name,
            case.zrot,
            case.yrot,
            case.hkl,
            case.position,
            case.wavelength,
            {},
        )


class _TestCubic(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)
        self.ubcalc.set_lattice("Cubic", 1)
        # self.B = I * 2 * pi


class TestCubicVertical(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(name, zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=0 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(4 * TORAD), 0, sin(4 * TORAD)),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=4 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=0,
                        phi=90 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=90 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Pos(
                        mu=0,
                        delta=97.46959231642,
                        nu=0,
                        eta=97.46959231642 / 2,
                        chi=86.18592516571 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001-->100",
                    (cos(86 * TORAD), 0, sin(86 * TORAD)),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=86 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            for case in cases:
                if case.name == name:
                    return case

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name, zrot, constraint"),
        itertools.product(
            [
                "100",
                "100-->001",
                "010",
                "001",
                "0.1 0 1.5",
                "001-->100",
            ],
            [0, 2, -2, 45, -45, 90, -90],
            [
                {"a_eq_b": True, "mu": 0, "nu": 0},
                {"psi": 90, "mu": 0, "nu": 0},
                {"a_eq_b": True, "mu": 0, "qaz": 90},
            ],
        ),
    )
    def test_pairs_various_zrot_and_yrot0(self, name, zrot, constraint, make_cases):
        self.constraints.asdict = constraint
        case = make_cases(name, zrot, 0)
        if name == "001":
            with pytest.raises(DiffcalcException):
                self.case_generator(case)
        else:
            self.case_generator(case)

    @pytest.mark.parametrize(
        ("name, zrot, yrot, constraint"),
        itertools.product(
            [
                "100",
                "100-->001",
                "010",
                "001",
                "0.1 0 1.5",
                "001-->100",
            ],
            [1, -1],
            [
                2,
            ],
            [
                {"a_eq_b": True, "mu": 0, "nu": 0},
                {"psi": 90, "mu": 0, "nu": 0},
                {"a_eq_b": True, "mu": 0, "qaz": 90},
            ],
        ),
    )
    def test_hkl_to_angles_zrot_yrot(self, name, zrot, yrot, constraint, make_cases):
        self.constraints.asdict = constraint
        case = make_cases(name, zrot, yrot)
        self.case_generator(case)

    @pytest.mark.parametrize(
        ("constraint"),
        [
            {"a_eq_b": True, "mu": 0, "nu": 0},
            {"psi": 90, "mu": 0, "nu": 0},
            {"a_eq_b": True, "mu": 0, "qaz": 90},
        ],
    )
    def testHklDeltaGreaterThan90(self, constraint):
        self.constraints.asdict = constraint
        wavelength = 1
        hkl = (0.1, 0, 1.5)
        pos = Pos(
            mu=0,
            delta=97.46959231642,
            nu=0,
            eta=97.46959231642 / 2,
            chi=86.18592516571,
            phi=0,
        )
        self._check_hkl_to_angles("", 0, 0, hkl, pos, wavelength)


class TestCubicVertical_ChiPhiMode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"nu": 0, "chi": 90.0, "phi": 0.0}
        self.places = 5

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "110",
                    (1, 1, 0),
                    Pos(
                        mu=-90,
                        delta=90,
                        nu=0,
                        eta=90,
                        chi=90,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (sin(4 * TORAD), 0, cos(4 * TORAD)),
                    Pos(
                        mu=-8.01966360660,
                        delta=60,
                        nu=0,
                        eta=29.75677306273,
                        chi=90,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=120,
                        chi=90,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30 - yrot,
                        chi=90,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Pos(
                        mu=-5.077064540005,
                        delta=97.46959231642,
                        nu=0,
                        eta=48.62310452627,
                        chi=90 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->001",
                    (0, cos(86 * TORAD), sin(86 * TORAD)),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=34,
                        chi=90,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name"), ["110", "100-->001", "010", "001", "0.1 0 1.5", "010-->001"]
    )
    def test_pairs_zrot0_yrot0(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestCubic_FixedPhiMode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"mu": 0, "nu": 0, "phi": 0}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(name, zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30 + zrot,
                        chi=0 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(4 * TORAD), 0, sin(4 * TORAD)),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30 + zrot,
                        chi=4 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                # Pair('010', (0, 1, 0),
                #     Pos(mu=0, delta=60, nu=0, eta=30 + self.zrot, chi=0, phi=90, unit='DEG')),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30 + zrot,
                        chi=90 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Pos(
                        mu=0,
                        delta=97.46959231642,
                        nu=0,
                        eta=97.46959231642 / 2 + zrot,
                        chi=86.18592516571 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001-->100",
                    (cos(86 * TORAD), 0, sin(86 * TORAD)),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30 + zrot,
                        chi=86 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            for case in cases:
                if case.name == name:
                    return case

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name, yrot"),
        itertools.product(
            ["100", "100-->001", "001", "0.1 0 1.5", "001-->100"],
            [0, 2, -2, 45, -45, 90, -90],
        ),
    )
    def test_pairs_various_zrot0_and_yrot(self, name, yrot, make_cases):
        case = make_cases(name, 0, yrot)
        self.case_generator(case)


class TestCubic_FixedPhi30Mode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"mu": 0, "nu": 0, "phi": 30}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=0 + zrot,
                        chi=0 - yrot,
                        phi=30,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                # Pair('100-->001', (cos(4 * TORAD), 0, sin(4 * TORAD)),
                #     Pos(mu=0, delta=60, nu=0, eta=0 + self.zrot, chi=4 - self.yrot,
                #         phi=30, unit='DEG'),),
                # Pair('010', (0, 1, 0),
                #     Pos(mu=0, delta=60, nu=0, eta=30 + self.zrot, chi=0, phi=90, unit='DEG')),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30 + zrot,
                        chi=90 - yrot,
                        phi=30,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Pos(
                        mu=0,
                        delta=97.46959231642,
                        nu=0,
                        eta=46.828815370173 + zrot,
                        chi=86.69569481984 - yrot,
                        phi=30,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                # Pair('001-->100', (cos(86 * TORAD), 0, sin(86 * TORAD)),
                #     Pos(mu=0, delta=60, nu=0, eta=0 + self.zrot, chi=86 - self.yrot,
                #         phi=30, unit='DEG')),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(("name"), ["100", "001", "0.1 0 1.5"])
    def test_pairs_zrot0_and_yrot0(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestCubic_FixedPhiMode010(TestCubic_FixedPhiMode):
    def setup_method(self):
        TestCubic_FixedPhiMode.setup_method(self)
        self.constraints.asdict = {"mu": 0, "nu": 0, "phi": 90}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30 + zrot,
                        chi=0,
                        phi=90,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(("yrot"), [0, 2, -2, 45, -45, 90, -90])
    def test_pairs_various_zrot0_and_yrot(self, yrot, make_cases):
        case = make_cases(0, yrot)
        self.case_generator(case["010"])


class TestCubicVertical_MuEtaMode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"nu": 0, "mu": 90.0, "eta": 0.0}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "011",
                    (0, 1, 1),
                    Pos(
                        mu=90,
                        delta=90,
                        nu=0,
                        eta=0,
                        chi=0,
                        phi=90,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (sin(4 * TORAD), 0, cos(4 * TORAD)),
                    Pos(
                        mu=90,
                        delta=60,
                        nu=0,
                        eta=0,
                        chi=56,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=90,
                        delta=60,
                        nu=0,
                        eta=0,
                        chi=-30,
                        phi=90,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=90,
                        delta=60,
                        nu=0,
                        eta=0,
                        chi=60,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Pos(
                        mu=90,
                        delta=97.46959231642,
                        nu=0,
                        eta=0,
                        chi=37.45112900750 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->001",
                    (0, cos(86 * TORAD), sin(86 * TORAD)),
                    Pos(
                        mu=90,
                        delta=60,
                        nu=0,
                        eta=0,
                        chi=56,
                        phi=90,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name"),
        [
            "011",
            "100-->001",
            "010",
            pytest.param("001", marks=pytest.mark.xfail(raises=DiffcalcException)),
            "0.1 0 1.5",
            "010-->001",
        ],
    )
    def test_pairs_zrot0_yrot0(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestCubic_FixedRefMuPhiMode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"psi": 90, "mu": 0, "phi": 0}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=0,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->100",
                    (sin(4 * TORAD), cos(4 * TORAD), 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=120 - 4,
                        chi=0,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=120,
                        chi=0,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=90,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Pos(
                        mu=0,
                        delta=97.46959231642,
                        nu=0,
                        eta=97.46959231642 / 2,
                        chi=86.18592516571,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001-->100",
                    (cos(86 * TORAD), 0, sin(86 * TORAD)),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=86,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name"),
        [
            "100",
            "010-->100",
            "010",
            pytest.param("001", marks=pytest.mark.xfail(raises=DiffcalcException)),
            "0.1 0 1.5",
            "001-->100",
        ],
    )
    def test_pairs_various_zrot0_and_yrot0(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestCubic_FixedRefEtaPhiMode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"psi": 0, "eta": 0, "phi": 0}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=-90,
                        delta=60,
                        nu=0,
                        eta=0,
                        chi=30,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(4 * TORAD), 0, sin(4 * TORAD)),
                    Pos(
                        mu=-90,
                        delta=60,
                        nu=0,
                        eta=0,
                        chi=30 + 4,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=120,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=180,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=90,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 0.15),  # cover case where delta > 90 !
                    Pos(
                        mu=-90,
                        delta=10.34318,
                        nu=0,
                        eta=0,
                        chi=61.48152,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->001",
                    (0, cos(4 * TORAD), sin(4 * TORAD)),
                    Pos(
                        mu=120 + 4,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=180,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name"),
        [
            "100",
            "100-->001",
            "010",
            pytest.param("001", marks=pytest.mark.xfail(raises=DiffcalcException)),
            "0.1 0 1.5",
            "010-->001",
        ],
    )
    def test_pairs_various_zrot0_and_yrot0(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestCubicVertical_Bisect(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"nu": 0, "bisect": True, "omega": 0}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "101",
                    (1, 0, 1),
                    Pos(
                        mu=0,
                        delta=90,
                        nu=0,
                        eta=45,
                        chi=45,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "10m1",
                    (1, 0, -1),
                    Pos(
                        mu=0,
                        delta=90,
                        nu=0,
                        eta=45,
                        chi=-45,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "011",
                    (0, 1, 1),
                    Pos(
                        mu=0,
                        delta=90,
                        nu=0,
                        eta=45,
                        chi=45,
                        phi=90,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (sin(4 * TORAD), 0, cos(4 * TORAD)),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=86,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=0,
                        phi=90,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=90,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Pos(
                        mu=0,
                        delta=97.46959231642,
                        nu=0,
                        eta=48.73480,
                        chi=86.18593 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->001",
                    (0, cos(86 * TORAD), sin(86 * TORAD)),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=86,
                        phi=90,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name"),
        [
            "101",
            "10m1",
            "011",
            "100-->001",
            "010",
            pytest.param("001", marks=pytest.mark.xfail(raises=DiffcalcException)),
            "0.1 0 1.5",
            "010-->001",
        ],
    )
    def test_pairs_zrot0_yrot0(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestCubicVertical_Bisect_NuMu(TestCubicVertical_Bisect):
    def setup_method(self):
        TestCubicVertical_Bisect.setup_method(self)
        self.constraints.asdict = {"nu": 0, "bisect": True, "mu": 0}


class TestCubicVertical_Bisect_qaz(TestCubicVertical_Bisect):
    def setup_method(self):
        TestCubicVertical_Bisect.setup_method(self)
        self.constraints.asdict = {"qaz": 90.0, "bisect": True, "mu": 0}


class TestCubicHorizontal_Bisect(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"delta": 0, "bisect": True, "omega": 0}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "101",
                    (1, 0, 1),
                    Pos(
                        mu=45,
                        delta=0,
                        nu=90,
                        eta=0,
                        chi=45,
                        phi=180,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "10m1",
                    (1, 0, -1),
                    Pos(
                        mu=45,
                        delta=0,
                        nu=90,
                        eta=0,
                        chi=135,
                        phi=180,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "011",
                    (0, 1, 1),
                    Pos(
                        mu=45,
                        delta=0,
                        nu=90,
                        eta=0,
                        chi=45,
                        phi=270,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (sin(4 * TORAD), 0, cos(4 * TORAD)),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=4,
                        phi=180,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=90,
                        phi=270,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Pos(
                        mu=48.73480,
                        delta=0,
                        nu=97.46959231642,
                        eta=0,
                        chi=3.81407 - yrot,
                        phi=180 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->001",
                    (0, cos(86 * TORAD), sin(86 * TORAD)),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=4,
                        phi=270,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name"),
        [
            "101",
            "10m1",
            "011",
            "100-->001",
            "010",
            pytest.param("001", marks=pytest.mark.xfail(raises=DiffcalcException)),
            "0.1 0 1.5",
            "010-->001",
        ],
    )
    def test_pairs_zrot0_yrot0(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestCubicHorizontal_Bisect_NuMu(TestCubicHorizontal_Bisect):
    def setup_method(self):
        TestCubicHorizontal_Bisect.setup_method(self)
        self.constraints.asdict = {"delta": 0, "bisect": True, "eta": 0}


class TestCubicHorizontal_Bisect_qaz(TestCubicHorizontal_Bisect):
    def setup_method(self):
        TestCubicHorizontal_Bisect.setup_method(self)
        self.constraints.asdict = {"qaz": 0, "bisect": True, "eta": 0}


class SkipTestHklCalculatorWithCubicMode_aeqb_delta_60(TestCubicVertical):
    """
    Works to 4-5 decimal places but chooses different solutions when phi||eta .
    Skip all tests.
    """

    def setup_method(self):
        TestCubicVertical.setup_method(self)
        self.constraints.asdict = {"a_eq_b": True, "mu": 0, "delta": 60}
        self.places = 5


class TestCubicHorizontal(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=90 + yrot,
                        phi=-180 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(4 * TORAD), 0, sin(4 * TORAD)),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=90 - 4 + yrot,
                        phi=-180 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=90,
                        phi=-90 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),  # degenrate case mu||phi
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001-->100",
                    (cos(86 * TORAD), 0, sin(86 * TORAD)),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0 - 4 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name, constraint"),
        itertools.product(
            [
                "100",
                "100-->001",
                "010",
                "001",
                "001-->100",
            ],
            [
                {"a_eq_b": True, "qaz": 0, "eta": 0},
                {"a_eq_b": True, "delta": 0, "eta": 0},
            ],
        ),
    )
    def test_pairs_zrot0_yrot0(self, name, constraint, make_cases):
        self.constraints.asdict = constraint
        case = make_cases(0, 0)
        if name in ("100", "010", "001"):
            with pytest.raises(DiffcalcException):
                self.case_generator(case[name])
        else:
            self.case_generator(case[name])

    #    def test_pairs_various_zrot_and_yrot0(self):
    #        for zrot in [0, 2, -2, 45, -45, 90, -90]:
    # -180, 180 work but with cut problem
    #            self.makes_cases(zrot, 0)
    #            self.case_dict['001'].fails = True # q||n
    #            for case_tuple in self.case_generator():
    #                yield case_tuple

    @pytest.mark.parametrize(
        ("name, zrot, yrot, constraint"),
        itertools.product(
            [
                "100",
                "100-->001",
                "010",
                "001",
                "001-->100",
            ],
            [
                1,
            ],
            [2, -2],
            [
                {"a_eq_b": True, "qaz": 0, "eta": 0},
                {"a_eq_b": True, "delta": 0, "eta": 0},
            ],
        ),
    )
    def test_hkl_to_angles_zrot_yrot(self, name, zrot, yrot, constraint, make_cases):
        self.constraints.asdict = constraint
        case = make_cases(zrot, yrot)
        if name == "010":
            with pytest.raises(DiffcalcException):
                self.case_generator(case[name])
        else:
            self.case_generator(case[name])


# class TestCubicHorizontal_qaz0_aeqb(_TestCubicHorizontal):
#    def setup_method(self):
#        _TestCubicHorizontal.setup_method(self)
#        self.constraints.asdict = {"a_eq_b": True, "qaz": 0, "eta": 0}
#
#
# class TestCubicHorizontal_delta0_aeqb(_TestCubicHorizontal):
#    def setup_method(self):
#        _TestCubicHorizontal.setup_method(self)
#        self.constraints.asdict = {"a_eq_b": True, "delta": 0, "eta": 0}


class TestCubic_FixedDeltaEtaPhi0Mode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"eta": 0, "delta": 0, "phi": 0}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=-90 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(4 * TORAD), 0, sin(4 * TORAD)),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=-90 + 4 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=120,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),  # degenerate case chi||q
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001-->100",
                    (cos(86 * TORAD), 0, sin(86 * TORAD)),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0 - 4 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name, yrot"),
        itertools.product(
            [
                "100",
                "100-->001",
                "010",
                "001",
                "001-->100",
            ],
            [0, 2, -2, 45, -45, 90, -90],
        ),
    )
    def test_pairs_various_zrot0_and_yrot(self, name, yrot, make_cases):
        case = make_cases(0, yrot)
        if name == "010":
            with pytest.raises(DiffcalcException):
                self.case_generator(case[name])
        else:
            self.case_generator(case[name])


class TestCubic_FixedDeltaEtaPhi30Mode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"eta": 0, "delta": 0, "phi": 30}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=0,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=-90 - yrot,
                        phi=30,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=90,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=-90 - yrot,
                        phi=30,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0 - yrot,
                        phi=30,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(("name"), ["100", "010", "001"])
    def test_pairs_various_zrot0_and_yrot(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestCubic_FixedDeltaEtaChi0Mode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"eta": 0, "delta": 0, "chi": 0}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=120,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0,
                        phi=-90 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(4 * TORAD), 0, sin(4 * TORAD)),
                    Pos(
                        mu=120 - 4,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0,
                        phi=-90 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=120,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0,
                        phi=zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),  # degenerate case phi||q
                Pair(
                    "001-->100",
                    (cos(86 * TORAD), 0, sin(86 * TORAD)),
                    Pos(
                        mu=30 - 4,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=0,
                        phi=90 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name, zrot"),
        itertools.product(
            [
                "100",
                "100-->001",
                "010",
                "001",
                "001-->100",
            ],
            [0, 2, -2],
        ),
    )
    def test_pairs_various_zrot_and_yrot0(self, name, zrot, make_cases):
        case = make_cases(zrot, 0)
        if name == "001":
            with pytest.raises(DiffcalcException):
                self.case_generator(case[name])
        else:
            self.case_generator(case[name])


class TestCubic_FixedDeltaEtaChi30Mode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"eta": 0, "delta": 0, "chi": 30}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=120,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=30,
                        phi=-90,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=120,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=30,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (-sin(30 * TORAD), 0, cos(30 * TORAD)),
                    Pos(
                        mu=30,
                        delta=0,
                        nu=60,
                        eta=0,
                        chi=30,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(("name"), ["100", "010", "100-->001"])
    def test_pairs_zrot0_and_yrot0(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestCubic_FixedGamMuChi90Mode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"mu": 0, "nu": 0, "chi": 90}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=120,
                        chi=90,
                        phi=-90 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->100",
                    (sin(4 * TORAD), cos(4 * TORAD), 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=120,
                        chi=90,
                        phi=-4 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=120,
                        chi=90,
                        phi=zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=30,
                        chi=90,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),  # degenerate case phi||q
                Pair(
                    "100-->010",
                    (sin(86 * TORAD), cos(86 * TORAD), 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=120,
                        chi=90,
                        phi=-90 + 4 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name, zrot"),
        itertools.product(
            [
                "100",
                "010-->100",
                "010",
                "001",
                "100-->010",
            ],
            [0, 2, -2],
        ),
    )
    def test_pairs_various_zrot_and_yrot0(self, name, zrot, make_cases):
        case = make_cases(0, zrot)
        if name == "001":
            with pytest.raises(DiffcalcException):
                self.case_generator(case[name])
        else:
            self.case_generator(case[name])


class TestCubic_FixedGamMuChi30Mode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"mu": 0, "nu": 0, "chi": 30}

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=120,
                        chi=30,
                        phi=-90,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=120,
                        chi=30,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->010",
                    (sin(30 * TORAD), cos(30 * TORAD), 0),
                    Pos(
                        mu=0,
                        delta=60,
                        nu=0,
                        eta=120,
                        chi=30,
                        phi=-30,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(("name"), ["100", "010", "100-->010"])
    def test_pairs_zrot0_and_yrot0(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestAgainstSpecSixcB16_270608(_BaseTest):
    """NOTE: copied from test.diffcalc.scenarios.session3"""

    def setup_method(self):
        _BaseTest.setup_method(self)

        self.constraints.asdict = {"a_eq_b": True, "mu": 0, "nu": 0}
        self.places = 2

    def _configure_ub(self):
        self.ubcalc.set_lattice("name", 3.8401, 5.43072)
        U = array(
            (
                (0.997161, -0.062217, 0.042420),
                (0.062542, 0.998022, -0.006371),
                (-0.041940, 0.009006, 0.999080),
            )
        )
        self.ubcalc.set_u(U)

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 1.24
            cases = (
                Pair(
                    "7_9_13",
                    (0.7, 0.9, 1.3),
                    Pos(
                        mu=0,
                        delta=27.352179,
                        nu=0,
                        eta=13.676090,
                        chi=37.774500,
                        phi=53.965500,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100",
                    (1, 0, 0),
                    Pos(
                        mu=0,
                        delta=18.580230,
                        nu=0,
                        eta=9.290115,
                        chi=-2.403500,
                        phi=3.589000,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Pos(
                        mu=0,
                        delta=18.580230,
                        nu=0,
                        eta=9.290115,
                        chi=0.516000,
                        phi=93.567000,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "110",
                    (1, 1, 0),
                    Pos(
                        mu=0,
                        delta=26.394192,
                        nu=0,
                        eta=13.197096,
                        chi=-1.334500,
                        phi=48.602000,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(("name"), ["7_9_13", "100", "010", "110"])
    def test_hkl_to_angles_given_UB(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestThreeTwoCircleForDiamondI06andI10(_BaseTest):
    """
    This is a three circle diffractometer with only delta and omega axes
    and a chi axis with limited range around 90. It is operated with phi
    fixed and can only reach reflections with l (or z) component.

    The data here is taken from an experiment performed on Diamonds I06
    beamline.
    """

    def setup_method(self):
        _BaseTest.setup_method(self)
        self.constraints.asdict = {"phi": -90, "nu": 0, "mu": 0}
        self.wavelength = 12.39842 / 1.650

    def _configure_ub(self):
        self.ubcalc.set_lattice("xtal", 5.34, 13.2)
        self.ubcalc.set_u(I)

    def testHkl001(self):
        hkl = (0, 0, 1)
        pos = Pos(
            mu=0,
            delta=33.07329403295449,
            nu=0,
            eta=16.536647016477247,
            chi=90,
            phi=-90,
        )
        self._check_angles_to_hkl("001", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("001", 999, 999, hkl, pos, self.wavelength, {})

    def testHkl100(self):
        hkl = (1, 0, 0)
        pos = Pos(
            mu=0,
            delta=89.42926563609406,
            nu=0,
            eta=134.71463281804702,
            chi=0,
            phi=-90,
        )
        self._check_angles_to_hkl("100", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("100", 999, 999, hkl, pos, self.wavelength, {})

    def testHkl101(self):
        hkl = (1, 0, 1)
        pos = Pos(
            mu=0,
            delta=98.74666191021282,
            nu=0,
            eta=117.347760720783,
            chi=90,
            phi=-90,
        )
        self._check_angles_to_hkl("101", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("101", 999, 999, hkl, pos, self.wavelength, {})


class TestThreeTwoCircleForDiamondI06andI10Horizontal(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)
        self.constraints.asdict = {"phi": -90, "delta": 0, "eta": 0}
        self.wavelength = 12.39842 / 1.650

    def _configure_ub(self):
        self.ubcalc.set_lattice("xtal", 5.34, 13.2)
        self.ubcalc.set_u(I)

    def testHkl001(self):
        hkl = (0, 0, 1)
        pos = Pos(
            mu=16.536647016477247,
            delta=0,
            nu=33.07329403295449,
            eta=0,
            chi=0,
            phi=-90,
        )
        self._check_angles_to_hkl("001", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("001", 999, 999, hkl, pos, self.wavelength, {})

    @pytest.mark.xfail(raises=DiffcalcException)  # q || chi
    def testHkl100(self):
        hkl = (1, 0, 0)
        pos = Pos(
            mu=134.71463281804702,
            delta=0,
            nu=89.42926563609406,
            eta=0,
            chi=0,
            phi=-90,
        )
        self._check_angles_to_hkl("100", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("100", 999, 999, hkl, pos, self.wavelength, {})

    def testHkl101(self):
        hkl = (1, 0, 1)
        pos = Pos(
            mu=117.347760720783,
            delta=0,
            nu=98.74666191021282,
            eta=0,
            chi=0,
            phi=-90,
        )
        self._check_angles_to_hkl("101", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("101", 999, 999, hkl, pos, self.wavelength, {})


class TestThreeTwoCircleForDiamondI06andI10ChiDeltaEta(
    TestThreeTwoCircleForDiamondI06andI10Horizontal
):
    def setup_method(self):
        _BaseTest.setup_method(self)
        self.constraints.asdict = {"phi": -90, "chi": 0, "delta": 0}
        self.wavelength = 12.39842 / 1.650

    @pytest.mark.xfail(raises=DiffcalcException)  # q || eta
    def testHkl001(self):
        hkl = (0, 0, 1)
        pos = Pos(
            mu=16.536647016477247,
            delta=0,
            nu=33.07329403295449,
            eta=0,
            chi=0,
            phi=-90,
        )
        self._check_angles_to_hkl("001", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("001", 999, 999, hkl, pos, self.wavelength, {})

    def testHkl100(self):
        hkl = (1, 0, 0)
        pos = Pos(
            mu=134.71463281804702,
            delta=0,
            nu=89.42926563609406,
            eta=0,
            chi=0,
            phi=-90,
        )
        self._check_angles_to_hkl("100", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("100", 999, 999, hkl, pos, self.wavelength, {})


class TestFixedChiPhiPsiMode_DiamondI07SurfaceNormalHorizontal(_TestCubic):
    """
    The data here is taken from an experiment performed on Diamonds I07
    beamline, obtained using Vlieg's DIF software"""

    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"chi": 0, "phi": 0, "a_eq_b": True}
        self.wavelength = 1

    def _configure_ub(self):
        self.ubcalc.set_u(I)
        # Set some random reference vector orientation
        # that won't coincide with the scattering vector direction.
        # self.mock_ubcalc.n_phi = np.array([[0.087867277], [0.906307787], [0.413383038]])
        # self.mock_ubcalc.n_phi = np.array([[0.], [1.], [0.]])

    @pytest.mark.parametrize(
        ("hkl", "pos", "places"),
        [
            pytest.param(
                (0, 0, 1),
                Pos(
                    mu=30,
                    delta=0,
                    nu=60,
                    eta=90,
                    chi=0,
                    phi=0,
                ),
                4,
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (0, 1, 0),  # betaout=0
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=120,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (0, 1, 1),  # betaout=30
                Pos(
                    mu=30,
                    delta=54.7356,
                    nu=90,
                    eta=125.2644,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 0, 0),  # betaout=0
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=30,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 0, 1),  # betaout=30
                Pos(
                    mu=30,
                    delta=54.7356,
                    nu=90,
                    eta=35.2644,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 1, 0),  # betaout=0
                Pos(
                    mu=0,
                    delta=90,
                    nu=0,
                    eta=90,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 1, 0.0001),  # betaout=0
                Pos(
                    mu=0.0029,
                    delta=89.9971,
                    nu=90.0058,
                    eta=90,
                    chi=0,
                    phi=0,
                ),
                3,
            ),
            (
                (1, 1, 1),  # betaout=30
                Pos(
                    mu=30,
                    delta=54.7356,
                    nu=150,
                    eta=99.7356,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (1.1, 0, 0),  # betaout=0
                Pos(
                    mu=0,
                    delta=66.7340,
                    nu=0,
                    eta=33.3670,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (0.9, 0, 0),  # betaout=0
                Pos(
                    mu=0,
                    delta=53.4874,
                    nu=0,
                    eta=26.7437,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (0.7, 0.8, 0.8),  # betaout=23.5782
                Pos(
                    mu=23.5782,
                    delta=59.9980,
                    nu=76.7037,
                    eta=84.2591,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (0.7, 0.8, 0.9),  # betaout=26.7437
                Pos(
                    mu=26.74368,
                    delta=58.6754,
                    nu=86.6919,
                    eta=85.3391,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (0.7, 0.8, 1),  # betaout=30
                Pos(
                    mu=30,
                    delta=57.0626,
                    nu=96.86590,
                    eta=86.6739,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
        ],
    )
    def testHKL(self, hkl, pos, places):
        self.places = places
        self._check_angles_to_hkl("", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("", 999, 999, hkl, pos, self.wavelength, {})

    # def testHkl001(self):
    #    self._check((0, 0, 1),  # betaout=30
    #                Pos(mu=30, delta=0, nu=60, eta=90, chi=0, phi=0, unit='DEG'))
    #
    # def testHkl010(self):
    #    self._check((0, 1, 0),  # betaout=0
    #                Pos(mu=0, delta=60, nu=0, eta=120, chi=0, phi=0, unit='DEG'))
    #
    # def testHkl011(self):
    #    self._check((0, 1, 1),  # betaout=30
    #                Pos(mu=30, delta=54.7356, nu=90, eta=125.2644, chi=0, phi=0, unit='DEG'))
    #
    # def testHkl100(self):
    #    self._check((1, 0, 0),  # betaout=0
    #                Pos(mu=0, delta=60, nu=0, eta=30, chi=0, phi=0, unit='DEG'))
    #
    # def testHkl101(self):
    #    self._check((1, 0, 1),  # betaout=30
    #                Pos(mu=30, delta=54.7356, nu=90, eta=35.2644, chi=0, phi=0, unit='DEG'))
    #
    # def testHkl110(self):
    #    self._check((1, 1, 0),  # betaout=0
    #                Pos(mu=0, delta=90, nu=0, eta=90, chi=0, phi=0, unit='DEG'))
    #
    # def testHkl11nearly0(self):
    #    self.places = 3
    #    self._check((1, 1, .0001),  # betaout=0
    #                Pos(mu=0.0029, delta=89.9971, nu=90.0058, eta=90, chi=0,
    #                  phi=0, unit='DEG'))
    #
    # def testHkl111(self):
    #    self._check((1, 1, 1),  # betaout=30
    #                Pos(mu=30, delta=54.7356, nu=150, eta=99.7356, chi=0, phi=0, unit='DEG'))
    #
    # def testHklover100(self):
    #    self._check((1.1, 0, 0),  # betaout=0
    #                Pos(mu=0, delta=66.7340, nu=0, eta=33.3670, chi=0, phi=0, unit='DEG'))
    #
    # def testHklunder100(self):
    #    self._check((.9, 0, 0),  # betaout=0
    #                Pos(mu=0, delta=53.4874, nu=0, eta=26.7437, chi=0, phi=0, unit='DEG'))
    #
    # def testHkl788(self):
    #    self._check((.7, .8, .8),  # betaout=23.5782
    #                Pos(mu=23.5782, delta=59.9980, nu=76.7037, eta=84.2591,
    #                  chi=0, phi=0, unit='DEG'))
    #
    # def testHkl789(self):
    #    self._check((.7, .8, .9),  # betaout=26.7437
    #                Pos(mu=26.74368, delta=58.6754, nu=86.6919, eta=85.3391,
    #                  chi=0, phi=0, unit='DEG'))
    #
    # def testHkl7810(self):
    #    self._check((.7, .8, 1),  # betaout=30
    #                Pos(mu=30, delta=57.0626, nu=96.86590, eta=86.6739, chi=0,
    #                  phi=0, unit='DEG'))


class SkipTestFixedChiPhiPsiModeSurfaceNormalVertical(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"chi": 90, "phi": 0, "a_eq_b": True}
        self.wavelength = 1
        self.UB = I * 2 * pi
        self.places = 4

    def _configure_ub(self):
        self.mock_ubcalc.UB = self.UB

    def _check(self, hkl, pos, virtual_expected={}, fails=False):
        self._check_angles_to_hkl(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )
        if fails:
            self._check_hkl_to_angles_fails(
                "", 999, 999, hkl, pos, self.wavelength, virtual_expected
            )
        else:
            self._check_hkl_to_angles(
                "", 999, 999, hkl, pos, self.wavelength, virtual_expected
            )

    def testHkl001(self):
        self._check(
            (0, 0, 1),  # betaout=30
            Pos(
                mu=30,
                delta=0,
                nu=60,
                eta=90,
                chi=0,
                phi=0,
            ),
            fails=True,
        )

    def testHkl010(self):
        self._check(
            (0, 1, 0),  # betaout=0
            Pos(
                mu=120,
                delta=0,
                nu=60,
                eta=0,
                chi=90,
                phi=0,
            ),
        )

    def testHkl011(self):
        self._check(
            (0, 1, 1),  # betaout=30
            Pos(
                mu=30,
                delta=54.7356,
                nu=90,
                eta=125.2644,
                chi=0,
                phi=0,
            ),
        )

    def testHkl100(self):
        self._check(
            (1, 0, 0),  # betaout=0
            Pos(
                mu=0,
                delta=60,
                nu=0,
                eta=30,
                chi=0,
                phi=0,
            ),
        )

    def testHkl101(self):
        self._check(
            (1, 0, 1),  # betaout=30
            Pos(
                mu=30,
                delta=54.7356,
                nu=90,
                eta=35.2644,
                chi=0,
                phi=0,
            ),
        )

    def testHkl110(self):
        self._check(
            (1, 1, 0),  # betaout=0
            Pos(
                mu=0,
                delta=90,
                nu=0,
                eta=90,
                chi=0,
                phi=0,
            ),
        )

    def testHkl11nearly0(self):
        self.places = 3
        self._check(
            (1, 1, 0.0001),  # betaout=0
            Pos(
                mu=0.0029,
                delta=89.9971,
                nu=90.0058,
                eta=90,
                chi=0,
                phi=0,
            ),
        )

    def testHkl111(self):
        self._check(
            (1, 1, 1),  # betaout=30
            Pos(
                mu=30,
                delta=54.7356,
                nu=150,
                eta=99.7356,
                chi=0,
                phi=0,
            ),
        )

    def testHklover100(self):
        self._check(
            (1.1, 0, 0),  # betaout=0
            Pos(
                mu=0,
                delta=66.7340,
                nu=0,
                eta=33.3670,
                chi=0,
                phi=0,
            ),
        )

    def testHklunder100(self):
        self._check(
            (0.9, 0, 0),  # betaout=0
            Pos(
                mu=0,
                delta=53.4874,
                nu=0,
                eta=26.7437,
                chi=0,
                phi=0,
            ),
        )

    def testHkl788(self):
        self._check(
            (0.7, 0.8, 0.8),  # betaout=23.5782
            Pos(
                mu=23.5782,
                delta=59.9980,
                nu=76.7037,
                eta=84.2591,
                chi=0,
                phi=0,
            ),
        )

    def testHkl789(self):
        self._check(
            (0.7, 0.8, 0.9),  # betaout=26.7437
            Pos(
                mu=26.74368,
                delta=58.6754,
                nu=86.6919,
                eta=85.3391,
                chi=0,
                phi=0,
            ),
        )

    def testHkl7810(self):
        self._check(
            (0.7, 0.8, 1),  # betaout=30
            Pos(
                mu=30,
                delta=57.0626,
                nu=96.86590,
                eta=86.6739,
                chi=0,
                phi=0,
            ),
        )


class SkipTestFixedChiPhiPsiModeSurfaceNormalVerticalI16(_TestCubic):
    # testing with Chris N. for pre christmas 2012 i16 experiment

    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"chi": 90, "phi": 0, "a_eq_b": True}
        self.wavelength = 1
        self.UB = I * 2 * pi
        self.places = 4

    def _configure_ub(self):
        self.mock_ubcalc.UB = self.UB

    def _check(self, hkl, pos, virtual_expected={}, fails=False):
        #        self._check_angles_to_hkl(
        #            '', 999, 999, hkl, pos, self.wavelength, virtual_expected)
        if fails:
            self._check_hkl_to_angles_fails(
                "", 999, 999, hkl, pos, self.wavelength, virtual_expected
            )
        else:
            self._check_hkl_to_angles(
                "", 999, 999, hkl, pos, self.wavelength, virtual_expected
            )

    def testHkl_1_1_0(self):
        self._check(
            (1, 1, 0.001),  # betaout=0
            Pos(
                mu=0,
                delta=60,
                nu=0,
                eta=30,
                chi=0,
                phi=0,
            ),
        )
        # (-89.9714,  89.9570,  90.0382,  90.0143,  90.0000,  0.0000)

    def testHkl_1_1_05(self):
        self._check(
            (1, 1, 0.5),  # betaout=0
            Pos(
                mu=0,
                delta=60,
                nu=0,
                eta=30,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_1_1_1(self):
        self._check(
            (1, 1, 1),  # betaout=0
            Pos(
                mu=0,
                delta=60,
                nu=0,
                eta=30,
                chi=0,
                phi=0,
            ),
        )
        #    (-58.6003,  42.7342,  132.9004,  106.3249,  90.0000,  0.0000

    def testHkl_1_1_15(self):
        self._check(
            (1, 1, 1.5),  # betaout=0
            Pos(
                mu=0,
                delta=60,
                nu=0,
                eta=30,
                chi=0,
                phi=0,
            ),
        )


class TestConstrain3Sample_ChiPhiEta(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"chi": 90, "phi": 0, "a_eq_b": True}
        self.wavelength = 1
        self.places = 4

    def _configure_ub(self):
        self.ubcalc.set_u(I)
        # Set some random reference vector orientation
        # that won't coincide with the scattering vector direction.
        # self.mock_ubcalc.n_phi = np.array([[0.087867277], [0.906307787], [0.413383038]])

    def _check(self, hkl, pos, virtual_expected={}):
        self._check_angles_to_hkl(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )
        self._check_hkl_to_angles(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )

    def testHkl_all0_001(self):
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0}
        self._check(
            (0, 0, 1),
            Pos(
                mu=30,
                delta=0,
                nu=60,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_all0_010(self):
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0}
        self._check(
            (0, 1, 0),
            Pos(
                mu=120,
                delta=0,
                nu=60,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_all0_011(self):
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0}
        self._check(
            (0, 1, 1),
            Pos(
                mu=90,
                delta=0,
                nu=90,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_phi30_100(self):
        self.constraints.asdict = {"chi": 0, "phi": 30, "eta": 0}
        self._check(
            (1, 0, 0),
            Pos(
                mu=0,
                delta=60,
                nu=0,
                eta=0,
                chi=0,
                phi=30,
            ),
        )

    def testHkl_eta30_100(self):
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 30}
        self._check(
            (1, 0, 0),
            Pos(
                mu=0,
                delta=60,
                nu=0,
                eta=30,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_phi90_110(self):
        self.constraints.asdict = {"chi": 0, "phi": 90, "eta": 0}
        self._check(
            (1, 1, 0),
            Pos(
                mu=0,
                delta=90,
                nu=0,
                eta=0,
                chi=0,
                phi=90,
            ),
        )

    def testHkl_eta90_110(self):
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 90}
        self._check(
            (1, 1, 0),
            Pos(
                mu=0,
                delta=90,
                nu=0,
                eta=90,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_all0_1(self):
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0}
        self._check(
            (0.01, 0.01, 0.1),
            Pos(
                mu=8.6194,
                delta=0.5730,
                nu=5.7607,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_all0_2(self):
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0 * TORAD}
        self._check(
            (0, 0, 0.1),
            Pos(
                mu=2.8660,
                delta=0,
                nu=5.7320,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_all0_3(self):
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0}
        self._check(
            (0.1, 0, 0.01),
            Pos(
                mu=30.3314,
                delta=5.7392,
                nu=0.4970,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_show_all_solutionsall0_3(self):
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0}
        self._check(
            (0.1, 0, 0.01),
            Pos(
                mu=30.3314,
                delta=5.7392,
                nu=0.4970,
                eta=0,
                chi=0,
                phi=0,
            ),
        )
        # print self.hklcalc.hkl_to_all_angles(.1, 0, .01, 1)

    def testHkl_all0_010to001(self):
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0}
        self._check(
            (0, cos(4 * TORAD), sin(4 * TORAD)),
            Pos(
                mu=120 - 4,
                delta=0,
                nu=60,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_1(self):
        self.wavelength = 0.1
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0}
        self._check(
            (0, 0, 1),
            Pos(
                mu=2.8660,
                delta=0,
                nu=5.7320,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_2(self):
        self.wavelength = 0.1
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0}
        self._check(
            (0, 0, 1),
            Pos(
                mu=2.8660,
                delta=0,
                nu=5.7320,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_3(self):
        self.wavelength = 0.1
        self.constraints.asdict = {"chi": 0, "phi": 0, "eta": 0}
        self._check(
            (1, 0, 0.1),
            Pos(
                mu=30.3314,
                delta=5.7392,
                nu=0.4970,
                eta=0,
                chi=0,
                phi=0,
            ),
        )


class TestConstrain3Sample_MuChiPhi(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"chi": 90, "phi": 0, "a_eq_b": True}
        self.wavelength = 1
        self.places = 4

    def _configure_ub(self):
        self.ubcalc.set_u(I)
        # Set some random reference vector orientation
        # that won't coincide with the scattering vector direction.
        # self.mock_ubcalc.n_phi = np.array([[0.087867277], [0.906307787], [0.413383038]])

    @pytest.mark.parametrize(
        ("hkl", "pos", "constraint"),
        [
            (
                (0, 0, 1),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=30,
                    chi=90,
                    phi=0,
                ),
                {"chi": 90, "phi": 0, "mu": 0},
            ),
            (
                (0, 1, 0),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=120,
                    chi=90,
                    phi=0,
                ),
                {"chi": 90, "phi": 0, "mu": 0},
            ),
            (
                (0, 1, 1),
                Pos(
                    mu=0,
                    delta=90,
                    nu=0,
                    eta=90,
                    chi=90,
                    phi=0,
                ),
                {"chi": 90, "phi": 0, "mu": 0},
            ),
            (
                (1, 0, 0),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=-60,
                    chi=90,
                    phi=90,
                ),
                {"chi": 90, "phi": 90, "mu": 0},
            ),
            pytest.param(
                (-1, 1, 0),
                Pos(
                    mu=90,
                    delta=90,
                    nu=0,
                    eta=90,
                    chi=90,
                    phi=0,
                ),
                {"chi": 90, "phi": 0, "mu": 90},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (1, 0, 1),
                Pos(
                    mu=0,
                    delta=90,
                    nu=0,
                    eta=0,
                    chi=90,
                    phi=90,
                ),
                {"chi": 90, "phi": 90, "mu": 0},
            ),
            (
                (sin(4 * TORAD), cos(4 * TORAD), 0),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=120 - 4,
                    chi=0,
                    phi=0,
                ),
                {"chi": 0, "phi": 0, "mu": 0},
            ),
        ],
    )
    def testHkl(self, hkl, pos, constraint):
        self.constraints.asdict = constraint
        self._check_angles_to_hkl("", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("", 999, 999, hkl, pos, self.wavelength, {})

    # def testHkl_all0_001(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'phi': 0, 'mu': 0}
    #    self._check((0, 0, 1),
    #                Pos(mu=0, delta=60, nu=0, eta=30, chi=90, phi=0, unit='DEG'))
    #
    # def testHkl_all0_010(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'phi': 0, 'mu': 0}
    #    self._check((0, 1, 0),
    #                Pos(mu=0, delta=60, nu=0, eta=120, chi=90, phi=0, unit='DEG'))
    #
    # def testHkl_all0_011(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'phi': 0, 'mu': 0}
    #    self._check((0, 1, 1),
    #                Pos(mu=0, delta=90, nu=0, eta=90, chi=90, phi=0, unit='DEG'))
    #
    # def testHkl_etam60_100(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'phi': 90 * TORAD, 'mu': 0}
    #    self._check((1, 0, 0),
    #                Pos(mu=0, delta=60, nu=0, eta=-60, chi=90, phi=90, unit='DEG'))
    #
    # def testHkl_eta90_m110(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'phi': 0, 'mu': 90 * TORAD}
    #    self._check((-1, 1, 0),
    #                Pos(mu=90, delta=90, nu=0, eta=90, chi=90, phi=0, unit='DEG'), fails=True)
    #
    # def testHkl_eta0_101(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'phi': 90 * TORAD, 'mu': 0}
    #    self._check((1, 0, 1),
    #                Pos(mu=0, delta=90, nu=0, eta=0, chi=90, phi=90, unit='DEG'))
    #
    # def testHkl_all0_010to100(self):
    #    self.constraints.asdict = {'chi': 0, 'phi': 0, 'mu': 0}
    #    self._check((sin(4 * TORAD), cos(4 * TORAD), 0),
    #                Pos(mu=0, delta=60, nu=0, eta=120 - 4, chi=0, phi=0, unit='DEG'))


class TestConstrain3Sample_MuEtaChi(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"chi": 90, "eta": 0, "a_eq_b": True}
        self.wavelength = 1
        self.UB = I * 2 * pi
        self.places = 4

    def _configure_ub(self):
        self.ubcalc.set_u(I)
        # Set some random reference vector orientation
        # that won't coincide with the scattering vector direction.
        # self.mock_ubcalc.n_phi = np.array([[0.087867277], [0.906307787], [0.413383038]])

    @pytest.mark.parametrize(
        ("hkl", "pos", "constraint"),
        [
            pytest.param(
                (0, 0, 1),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=30,
                    chi=90,
                    phi=0,
                ),
                {"eta": 30, "chi": 90, "mu": 0},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (0, 1, 0),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=120,
                    chi=90,
                    phi=0,
                ),
                {"chi": 90, "eta": 120, "mu": 0},
            ),
            pytest.param(
                (0, 1, 1),
                Pos(
                    mu=0,
                    delta=90,
                    nu=0,
                    eta=90,
                    chi=90,
                    phi=0,
                ),
                {"chi": 90, "eta": 90, "mu": 0},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (1, 0, 0),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=-60,
                    chi=90,
                    phi=90,
                ),
                {"chi": 90, "eta": -60, "mu": 0},
            ),
            (
                (-1, 1, 0),
                Pos(
                    mu=90,
                    delta=90,
                    nu=0,
                    eta=90,
                    chi=90,
                    phi=0,
                ),
                {"chi": 90, "eta": 90, "mu": 90},
            ),
            (
                (1, 0, 1),
                Pos(
                    mu=0,
                    delta=90,
                    nu=0,
                    eta=0,
                    chi=90,
                    phi=90,
                ),
                {"chi": 90, "eta": 0, "mu": 0},
            ),
            (
                (sin(4 * TORAD), cos(4 * TORAD), 0),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=0,
                    chi=0,
                    phi=120 - 4,
                ),
                {"chi": 0, "eta": 0, "mu": 0},
            ),
        ],
    )
    def _check(self, hkl, pos, constraint):
        self.constraints.asdict = constraint
        self._check_angles_to_hkl("", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("", 999, 999, hkl, pos, self.wavelength, {})

    # def testHkl_all0_001(self):
    #    self.constraints.asdict = {'eta': 30 * TORAD, 'chi': 90 * TORAD, 'mu': 0}
    #    self._check((0, 0, 1),
    #                Pos(mu=0, delta=60, nu=0, eta=30, chi=90, phi=0, unit='DEG'), fails=True)
    #
    # def testHkl_all0_010(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'eta': 120 * TORAD, 'mu': 0}
    #    self._check((0, 1, 0),
    #                Pos(mu=0, delta=60, nu=0, eta=120, chi=90, phi=0, unit='DEG'))
    #
    # def testHkl_all0_011(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'eta': 90 * TORAD, 'mu': 0}
    #    self._check((0, 1, 1),
    #                Pos(mu=0, delta=90, nu=0, eta=90, chi=90, phi=0, unit='DEG'), fails=True)
    #
    # def testHkl_phi90_100(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'eta': -60 * TORAD, 'mu': 0}
    #    self._check((1, 0, 0),
    #                Pos(mu=0, delta=60, nu=0, eta=-60, chi=90, phi=90, unit='DEG'))
    #
    # def testHkl_phi0_m110(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'eta': 90 * TORAD, 'mu': 90 * TORAD}
    #    self._check((-1, 1, 0),
    #                Pos(mu=90, delta=90, nu=0, eta=90, chi=90, phi=0, unit='DEG'))
    #
    # def testHkl_phi90_101(self):
    #    self.constraints.asdict = {'chi': 90 * TORAD, 'eta': 0, 'mu': 0}
    #    self._check((1, 0, 1),
    #                Pos(mu=0, delta=90, nu=0, eta=0, chi=90, phi=90, unit='DEG'))
    #
    # def testHkl_all0_010to100(self):
    #    self.constraints.asdict = {'chi': 0, 'eta': 0, 'mu': 0}
    #    self._check((sin(4 * TORAD), cos(4 * TORAD), 0),
    #                Pos(mu=0, delta=60, nu=0, eta=0, chi=0, phi=120 - 4, unit='DEG'))


class TestConstrain3Sample_MuEtaPhi(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.constraints.asdict = {"chi": 90, "phi": 0, "a_eq_b": True}
        self.wavelength = 1
        self.places = 4

    def _configure_ub(self):
        self.ubcalc.set_u(I)

    @pytest.mark.parametrize(
        ("hkl", "pos", "constraint"),
        [
            (
                (0, 0, 1),
                Pos(
                    mu=30,
                    delta=0,
                    nu=60,
                    eta=0,
                    chi=0,
                    phi=0,
                ),
                {"eta": 0, "phi": 0, "mu": 30},
            ),
            pytest.param(
                (0, 1, 0),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=120,
                    chi=90,
                    phi=0,
                ),
                {"eta": 120, "phi": 0, "mu": 0},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (0, 1, 1),
                Pos(
                    mu=0,
                    delta=90,
                    nu=0,
                    eta=90,
                    chi=90,
                    phi=0,
                ),
                {"eta": 90, "phi": 0, "mu": 0},
            ),
            pytest.param(
                (1, 0, 0),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=-60,
                    chi=90,
                    phi=90,
                ),
                {"eta": -60, "phi": 90, "mu": 0},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (-1, 1, 0),
                Pos(
                    mu=90,
                    delta=90,
                    nu=0,
                    eta=90,
                    chi=90,
                    phi=0,
                ),
                {"eta": 90, "phi": 0, "mu": 90},
            ),
            pytest.param(
                (1, 0, 1),
                Pos(
                    mu=0,
                    delta=90,
                    nu=0,
                    eta=0,
                    chi=90,
                    phi=90,
                ),
                {"eta": 0, "phi": 90, "mu": 0},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (sin(4 * TORAD), 0, cos(4 * TORAD)),
                Pos(
                    mu=0,
                    delta=60,
                    nu=0,
                    eta=30,
                    chi=90 - 4,
                    phi=0,
                ),
                {"eta": 30, "phi": 0, "mu": 0},
            ),
        ],
    )
    def testHKL(self, hkl, pos, constraint):
        self.constraints.asdict = constraint
        self._check_angles_to_hkl("", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("", 999, 999, hkl, pos, self.wavelength, {})

    # def testHkl_all0_001(self):
    #    self.constraints.asdict = {'eta': 0, 'phi': 0, 'mu': 30 * TORAD}
    #    self._check((0, 0, 1),
    #                Pos(mu=30, delta=0, nu=60, eta=0, chi=0, phi=0, unit='DEG'))
    #
    # def testHkl_all0_010(self):
    #    self.constraints.asdict = {'eta': 120 * TORAD, 'phi': 0, 'mu': 0}
    #    self._check((0, 1, 0),
    #                Pos(mu=0, delta=60, nu=0, eta=120, chi=90, phi=0, unit='DEG'), fails=True)
    #
    # def testHkl_all0_011(self):
    #    self.constraints.asdict = {'eta': 90 * TORAD, 'phi': 0, 'mu': 0}
    #    self._check((0, 1, 1),
    #                Pos(mu=0, delta=90, nu=0, eta=90, chi=90, phi=0, unit='DEG'))
    #
    # def testHkl_chi90_100(self):
    #    self.constraints.asdict = {'eta': -60 * TORAD, 'phi': 90 * TORAD, 'mu': 0}
    #    self._check((1, 0, 0),
    #                Pos(mu=0, delta=60, nu=0, eta=-60, chi=90, phi=90, unit='DEG'), fails=True)
    #
    # def testHkl_chi90_m110(self):
    #    self.constraints.asdict = {'eta': 90 * TORAD, 'phi': 0, 'mu': 90 * TORAD}
    #    self._check((-1, 1, 0),
    #                Pos(mu=90, delta=90, nu=0, eta=90, chi=90, phi=0, unit='DEG'))
    #
    # def testHkl_chi90_101(self):
    #    self.constraints.asdict = {'eta': 0, 'phi': 90 * TORAD, 'mu': 0}
    #    self._check((1, 0, 1),
    #                Pos(mu=0, delta=90, nu=0, eta=0, chi=90, phi=90, unit='DEG'), fails=True)
    #
    # def testHkl_all0_010to100(self):
    #    self.constraints.asdict = {'eta': 30 * TORAD, 'phi': 0, 'mu': 0}
    #    self._check((sin(4 * TORAD), 0, cos(4 * TORAD)),
    #                Pos(mu=0, delta=60, nu=0, eta=30, chi=90 - 4, phi=0, unit='DEG'))


class TestHorizontalDeltaNadeta0_JiraI16_32_failure(_BaseTest):
    """
    The data here is taken from a trial experiment which failed. Diamond's internal Jira:
    http://jira.diamond.ac.uk/browse/I16-32"""

    def setup_method(self):
        _BaseTest.setup_method(self)

        self.wavelength = 12.39842 / 8
        # self.UB = array(
        #    [
        #        [-1.46410390e00, -1.07335571e00, 2.44799214e-03],
        #        [3.94098508e-01, -1.07091934e00, -6.41132943e-04],
        #        [7.93297438e-03, 4.01315826e-03, 4.83650166e-01],
        #    ]
        # )
        self.places = 3

    def _configure_ub(self):
        self.ubcalc.set_lattice("I16_test", "Hexagonal", 4.785, 12.991)

        U = array(
            [
                [-9.65616334e-01, -2.59922060e-01, 5.06142415e-03],
                [2.59918682e-01, -9.65629598e-01, -1.32559487e-03],
                [5.23201232e-03, 3.55426382e-05, 9.99986312e-01],
            ]
        )
        self.ubcalc.set_u(U)

    def _check(self, hkl, pos, virtual_expected={}, skip_test_pair_verification=False):
        if not skip_test_pair_verification:
            self._check_angles_to_hkl(
                "", 999, 999, hkl, pos, self.wavelength, virtual_expected
            )
        self._check_hkl_to_angles(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )

    def test_hkl_bisecting_works_okay_on_i16(self):
        self.constraints.asdict = {"delta": 0, "a_eq_b": True, "eta": 0}
        self._check(
            [-1.1812112493619709, -0.71251524866987204, 5.1997083010199221],
            Pos(
                mu=26,
                delta=0,
                nu=52,
                eta=0,
                chi=45.2453,
                phi=186.6933 - 360,
            ),
        )

    def test_hkl_psi90_works_okay_on_i16(self):
        # This is failing here but on the live one. Suggesting some extreme sensitivity?
        self.constraints.asdict = {"delta": 0, "psi": -90, "eta": 0}
        self._check(
            [-1.1812112493619709, -0.71251524866987204, 5.1997083010199221],
            Pos(
                mu=26,
                delta=0,
                nu=52,
                eta=0,
                chi=45.2453,
                phi=186.6933 - 360,
            ),
        )

    def test_hkl_alpha_17_9776_used_to_fail(self):
        # This is failing here but on the live one. Suggesting some extreme sensitivity?
        self.constraints.asdict = {"delta": 0, "alpha": 17.9776, "eta": 0}
        self._check(
            [-1.1812112493619709, -0.71251524866987204, 5.1997083010199221],
            Pos(
                mu=26,
                delta=0,
                nu=52,
                eta=0,
                chi=45.2453,
                phi=186.6933 - 360,
            ),
        )

    def test_hkl_alpha_17_9776_failing_after_bigger_small(self):
        # This is failing here but on the live one. Suggesting some extreme sensitivity?
        self.constraints.asdict = {"delta": 0, "alpha": 17.8776, "eta": 0}
        self._check(
            [-1.1812112493619709, -0.71251524866987204, 5.1997083010199221],
            Pos(
                mu=25.85,
                delta=0,
                nu=52,
                eta=0,
                chi=45.2453,
                phi=-173.518,
            ),
        )


# skip_test_pair_verification


class TestAnglesToHkl_I16Examples:
    def setup_method(self):
        self.ubcalc = UBCalculation()
        self.constraints = Constraints()

        U = array(
            (
                (0.9996954135095477, -0.01745240643728364, -0.017449748351250637),
                (0.01744974835125045, 0.9998476951563913, -0.0003045864904520898),
                (0.017452406437283505, -1.1135499981271473e-16, 0.9998476951563912),
            )
        )
        self.WL1 = 1  # Angstrom
        self.ubcalc.set_lattice("Cubic", 1)
        self.ubcalc.set_u(U)

        self.hklcalc = HklCalculation(self.ubcalc, self.constraints)

    def test_anglesToHkl_mu_0_gam_0(self):
        pos = PosFromI16sEuler(1, 1, 30, 0, 60, 0)
        arrayeq_(self.hklcalc.get_hkl(pos, self.WL1), [1, 0, 0])

    def test_anglesToHkl_mu_0_gam_10(self):
        pos = PosFromI16sEuler(1, 1, 30, 0, 60, 10)
        arrayeq_(
            self.hklcalc.get_hkl(pos, self.WL1),
            [1.00379806, -0.006578435, 0.08682408],
        )

    def test_anglesToHkl_mu_10_gam_0(self):
        pos = PosFromI16sEuler(1, 1, 30, 10, 60, 0)
        arrayeq_(
            self.hklcalc.get_hkl(pos, self.WL1),
            [0.99620193, 0.0065784359, 0.08682408],
        )

    def test_anglesToHkl_arbitrary(self):
        pos = PosFromI16sEuler(1.9, 2.9, 30.9, 0.9, 60.9, 2.9)
        arrayeq_(
            self.hklcalc.get_hkl(pos, self.WL1),
            [1.01174189, 0.02368622, 0.06627361],
        )


class TestAnglesToHkl_I16Numerical(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)

        # self.UB = array(((1.11143, 0, 0), (0, 1.11143, 0), (0, 0, 1.11143)))

        self.constraints.asdict = {"mu": 0, "nu": 0, "phi": 0}
        self.wavelength = 1.0
        self.places = 6

    def _configure_ub(self):
        self.ubcalc.set_lattice("xtal", 5.653244295348863)
        self.ubcalc.set_u(I)
        self.n_phi = (0, 0, 1)

    def _check(
        self,
        testname,
        hkl,
        pos,
        virtual_expected={},
        skip_test_pair_verification=False,
    ):
        if not skip_test_pair_verification:
            self._check_angles_to_hkl(
                testname, 999, 999, hkl, pos, self.wavelength, virtual_expected
            )
        self._check_hkl_to_angles(
            testname, 999, 999, hkl, pos, self.wavelength, virtual_expected
        )

    def test_hkl_to_angles_given_UB(self):
        self._check(
            "I16_numeric",
            [2, 0, 0.000001],
            PosFromI16sEuler(0, 0.000029, 10.188639, 0, 20.377277, 0),
        )
        self._check(
            "I16_numeric",
            [2, 0.000001, 0],
            PosFromI16sEuler(0, 0, 10.188667, 0, 20.377277, 0),
        )


class TestAnglesToHkl_I16GaAsExample(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)

        # self.UB = array(
        #    (
        #        (-0.78935, 0.78234, 0.01191),
        #        (-0.44391, -0.46172, 0.90831),
        #        (0.64431, 0.64034, 0.64039),
        #    )
        # )

        self.constraints.asdict = {
            "qaz": 90.0,
            "alpha": 11.0,
            "mu": 0.0,
        }
        self.wavelength = 1.239842
        self.places = 3

    def _configure_ub(self):
        self.ubcalc.set_lattice("xtal", 5.65315)  # 5.6325) #3244295348863)
        U = array(
            [
                [-0.71021455, 0.70390373, 0.01071626],
                [-0.39940627, -0.41542895, 0.81724747],
                [0.57971538, 0.5761409, 0.57618724],
            ]
        )
        self.ubcalc.set_u(U)

    def _check(self, hkl, pos, virtual_expected={}, skip_test_pair_verification=False):
        if not skip_test_pair_verification:
            self._check_angles_to_hkl(
                "", 999, 999, hkl, pos, self.wavelength, virtual_expected
            )
        self._check_hkl_to_angles(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )

    def test_hkl_to_angles_given_UB(self):
        self._check(
            [1.0, 1.0, 1.0],
            PosFromI16sEuler(10.7263, 89.8419, 11.0000, 0.0, 21.8976, 0.0),
        )
        self._check(
            [0.0, 0.0, 2.0],
            PosFromI16sEuler(81.2389, 35.4478, 19.2083, 0.0, 25.3375, 0.0),
        )


class Test_I21ExamplesUB(_BaseTest):
    """NOTE: copied from test.diffcalc.scenarios.session3"""

    def setup_method(self):
        _BaseTest.setup_method(self)

        # self.constraints = Constraints()
        # self.hklcalc = HklCalculation(self.ubcalc, self.constraints)

        # B = array(((1.66222, 0.0, 0.0), (0.0, 1.66222, 0.0), (0.0, 0.0, 0.31260)))

        self.constraints.asdict = {"psi": 10, "mu": 0, "nu": 0}
        self.places = 5

    def _configure_ub(self):
        self.ubcalc.set_lattice("xtal", 3.78, 20.10)
        U = array(((1.0, 0.0, 0.0), (0.0, 0.18482, -0.98277), (0.0, 0.98277, 0.18482)))
        self.ubcalc.set_u(U)
        self.ubcalc.n_phi = (0, 0, 1)

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 12.39842 / 0.650
            cases = (
                Pair(
                    "0_0.2_0.25",
                    (0.0, 0.2, 0.25),
                    Pos(
                        mu=0,
                        delta=62.44607,
                        nu=0,
                        eta=28.68407,
                        chi=90.0 - 0.44753,
                        phi=-9.99008,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.25_0.2_0.1",
                    (0.25, 0.2, 0.1),
                    Pos(
                        mu=0,
                        delta=108.03033,
                        nu=0,
                        eta=3.03132,
                        chi=90 - 7.80099,
                        phi=87.95201,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(("name"), ["0_0.2_0.25", "0.25_0.2_0.1"])
    def test_hkl_to_angles_given_UB(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class SkipTest_FixedAlphaMuChiSurfaceNormalHorizontal(_BaseTest):
    """NOTE: copied from test.diffcalc.scenarios.session3"""

    def setup_method(self):
        _BaseTest.setup_method(self)

        U = array(
            (
                (-0.71022, 0.70390, 0.01071),
                (-0.39941, -0.41543, 0.81725),
                (0.57971, 0.57615, 0.57618),
            )
        )

        B = array(((1.11143, 0.0, 0.0), (0.0, 1.11143, 0.0), (0.0, 0.0, 1.11143)))

        self.UB = U @ B

        self.constraints.asdict = {"alpha": 12.0, "mu": 0, "chi": 0}
        self.places = 5

    def _configure_ub(self):
        self.mock_ubcalc.UB = self.UB
        self.mock_ubcalc.n_phi = array([[0], [0], [1]])

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 1.0
            cases = (
                Pair(
                    "0_0_0.25",
                    (2.0, 2.0, 0.0),
                    Pos(
                        mu=0,
                        delta=79.85393,
                        nu=0,
                        eta=39.92540,
                        chi=90.0,
                        phi=0.0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                # Pair('0.25_0.25_0', (0.25, 0.25, 0.0),
                #     Pos(mu=0, delta=27.352179, nu=0, eta=13.676090,
                #         chi=37.774500, phi=53.965500, unit='DEG')),
            )
            case_dict = {}
            for case in cases:
                case_dict[case.name] = case
            return case_dict

        return __make_cases_fixture

    @pytest.mark.parametrize(
        ("name"),
        [
            "0_0_0.25",
        ],
    )
    def test_hkl_to_angles_given_UB(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])
