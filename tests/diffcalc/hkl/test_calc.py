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


### ROTATIONS SHOULD ALSO BE IN RAD

import itertools
from itertools import chain
from math import cos, pi, radians, sin, sqrt

import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.util import DiffcalcException, I, y_rotation, z_rotation
from numpy import array

from tests.diffcalc.scenarios import PosFromI16sEuler
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


def test_str():
    hklcalc = HklCalculation("test_str")

    hklcalc.ubcalc.n_phi = (0, 0, 1)
    hklcalc.ubcalc.surf_nphi = (0, 0, 1)
    hklcalc.ubcalc.set_lattice("xtal", [1.0], "Cubic")
    hklcalc.ubcalc.add_reflection(
        (0, 0, 1), Position(0, pi / 3, 0, pi / 6, 0, 0), 12.4, "ref1"
    )
    hklcalc.ubcalc.add_orientation(
        (0, 1, 0), (0, 1, 0), Position(radians(1), 0, 0, 0, radians(2), 0), "orient1"
    )
    hklcalc.ubcalc.set_u(I)

    hklcalc.constraints = Constraints({"nu": 0.0, "psi": pi / 2, "phi": pi / 2})
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
    psi  : 1.5708
    phi  : 1.5708
"""
    )


class _BaseTest:
    def setup_method(self):

        self.hklcalc = HklCalculation("test")
        self.hklcalc.ubcalc.n_phi = (0, 0, 1)
        self.hklcalc.ubcalc.surf_nphi = (0, 0, 1)

        self.places = 5

    def _configure_ub(self):
        ZROT = z_rotation(self.zrot)  # -PHI
        YROT = y_rotation(self.yrot)  # +CHI
        U = ZROT @ YROT
        # UB = U @ self.B
        self.hklcalc.ubcalc.set_u(U)  # self.mock_ubcalc.UB = UB

    def _check_hkl_to_angles(
        self, testname, zrot, yrot, hkl, pos_expected, wavelength, virtual_expected={}
    ):
        print(
            "_check_hkl_to_angles(%s, %.1f, %.1f, %s, %s, %.2f, %s)"
            % (testname, zrot, yrot, hkl, pos_expected, wavelength, virtual_expected)
        )
        self.zrot, self.yrot = zrot, yrot
        self._configure_ub()

        pos_virtual_angles_pairs = self.hklcalc.get_position(
            hkl[0], hkl[1], hkl[2], wavelength
        )
        pos = list(chain(*pos_virtual_angles_pairs))[::2]
        virtual = list(chain(*pos_virtual_angles_pairs))[1::2]
        assert_array_almost_equal_in_list(
            pos_expected.astuple,
            [p.astuple for p in pos],
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
        self.hklcalc.ubcalc.set_lattice("Cubic", [1.0])
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
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=0 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(radians(4)), 0, sin(radians(4))),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=radians(4) - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=0,
                        phi=pi / 2 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=pi / 2 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Position(
                        mu=0,
                        delta=radians(97.46959231642),
                        nu=0,
                        eta=radians(97.46959231642 / 2),
                        chi=radians(86.18592516571) - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001-->100",
                    (cos(radians(86)), 0, sin(radians(86))),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=radians(86) - yrot,
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
        ("name", "zrot"),
        itertools.product(
            [
                "100",
                "100-->001",
                "010",
                "001",
                "001-->100",
            ],
            [0, radians(2), radians(-2), pi / 4, -pi / 4, pi / 2, -pi / 2],
        ),
    )
    def test_delta_aeqb_mu_zrot_and_yrot0(self, name, zrot, make_cases):
        self.hklcalc.constraints = Constraints(
            {"delta": pi / 3, "a_eq_b": True, "mu": 0}
        )
        case = make_cases(name, zrot, 0)
        if name == "001":
            with pytest.raises(DiffcalcException):
                self.case_generator(case)
        else:
            self.case_generator(case)

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
            [0, radians(2), radians(-2), pi / 4, -pi / 4, pi / 2, -pi / 2],
            [
                {"a_eq_b": True, "mu": 0, "nu": 0},
                {"psi": pi / 2, "mu": 0, "nu": 0},
                {"a_eq_b": True, "mu": 0, "qaz": pi / 2},
            ],
        ),
    )
    def test_pairs_various_zrot_and_yrot0(self, name, zrot, constraint, make_cases):
        self.hklcalc.constraints = Constraints(constraint)
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
            [radians(1), radians(-1)],
            [
                radians(2),
            ],
            [
                {"a_eq_b": True, "mu": 0, "nu": 0},
                {"psi": pi / 2, "mu": 0, "nu": 0},
                {"a_eq_b": True, "mu": 0, "qaz": pi / 2},
            ],
        ),
    )
    def test_hkl_to_angles_zrot_yrot(self, name, zrot, yrot, constraint, make_cases):
        self.hklcalc.constraints = Constraints(constraint)
        case = make_cases(name, zrot, yrot)
        self.case_generator(case)

    @pytest.mark.parametrize(
        ("constraint"),
        [
            {"a_eq_b": True, "mu": 0, "nu": 0},
            {"psi": pi / 2, "mu": 0, "nu": 0},
            {"a_eq_b": True, "mu": 0, "qaz": pi / 2},
        ],
    )
    def testHklDeltaGreaterThan90(self, constraint):
        self.hklcalc.constraints = Constraints(constraint)
        wavelength = 1
        hkl = (0.1, 0, 1.5)
        pos = Position(
            mu=0,
            delta=radians(97.46959231642),
            nu=0,
            eta=radians(97.46959231642 / 2),
            chi=radians(86.18592516571),
            phi=0,
        )
        self._check_hkl_to_angles("", 0, 0, hkl, pos, wavelength)


class TestCubicVertical_alpha90(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.ubcalc.n_hkl = (1, -1, 0)

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(name, zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "sqrt(2)00",
                    (sqrt(2), 0, 0),
                    Position(
                        mu=0,
                        delta=pi / 2,
                        nu=0,
                        eta=pi / 4,
                        chi=0,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "00sqrt(2)",
                    (0, 0, sqrt(2)),
                    Position(
                        mu=pi / 2,
                        delta=pi / 2,
                        nu=0,
                        eta=0,
                        chi=pi / 4,
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
        ("name", "zrot", "constraint"),
        itertools.product(
            [
                "sqrt(2)00",
            ],
            [
                0,
            ],
            [
                {"qaz": pi / 2, "alpha": pi / 2, "phi": 0},
                {"delta": pi / 2, "beta": 0, "phi": 0},
                {"delta": pi / 2, "betain": 0, "phi": 0},
            ],
        ),
    )
    def test_delta_alpha_mu_zrot_and_yrot0(self, name, zrot, constraint, make_cases):
        self.hklcalc.constraints = Constraints(constraint)
        case = make_cases(name, zrot, 0)
        self.case_generator(case)


class TestCubicVertical_ttheta180(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints({"nu": 0, "chi": 0, "phi": 0})
        self.hklcalc.ubcalc.n_hkl = (1, -1, 0)

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(name, zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "200",
                    (2, 0, 0),
                    Position(
                        mu=0,
                        delta=pi,
                        nu=0,
                        eta=pi / 2,
                        chi=0,
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
        ("name", "zrot"),
        itertools.product(
            [
                "200",
            ],
            [
                0,
            ],
        ),
    )
    def test_nu_chi_phi_0(self, name, zrot, make_cases):
        case = make_cases(name, zrot, 0)
        self.case_generator(case)


class TestCubicVertical_ChiPhiMode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints({"nu": 0, "chi": pi / 2, "phi": 0.0})
        self.places = 5

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "110",
                    (1, 1, 0),
                    Position(
                        mu=-pi / 2,
                        delta=pi / 2,
                        nu=0,
                        eta=pi / 2,
                        chi=pi / 2,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (sin(radians(4)), 0, cos(radians(4))),
                    Position(
                        mu=radians(-8.01966360660),
                        delta=pi / 3,
                        nu=0,
                        eta=radians(29.75677306273),
                        chi=pi / 2,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi * 2 / 3,
                        chi=pi / 2,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6 - yrot,
                        chi=pi / 2,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Position(
                        mu=radians(-5.077064540005),
                        delta=radians(97.46959231642),
                        nu=0,
                        eta=radians(48.62310452627),
                        chi=pi / 2 - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->001",
                    (0, cos(radians(86)), sin(radians(86))),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=radians(34),
                        chi=pi / 2,
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
        self.hklcalc.constraints = Constraints({"mu": 0, "nu": 0, "phi": 0})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(name, zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6 + zrot,
                        chi=0 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(radians(4)), 0, sin(radians(4))),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6 + zrot,
                        chi=radians(4) - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                # Pair('010', (0, 1, 0),
                #     Position(mu=0, delta=60, nu=0, eta=30 + self.zrot, chi=0, phi=90, unit='DEG')),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6 + zrot,
                        chi=pi / 2 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Position(
                        mu=0,
                        delta=radians(97.46959231642),
                        nu=0,
                        eta=radians(97.46959231642 / 2) + zrot,
                        chi=radians(86.18592516571) - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001-->100",
                    (cos(radians(86)), 0, sin(radians(86))),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6 + zrot,
                        chi=radians(86) - yrot,
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
            [0, radians(2), radians(-2), pi / 4, -pi / 4, pi / 2, -pi / 2],
        ),
    )
    def test_pairs_various_zrot0_and_yrot(self, name, yrot, make_cases):
        case = make_cases(name, 0, yrot)
        self.case_generator(case)


class TestCubic_FixedPhi30Mode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints({"mu": 0, "nu": 0, "phi": pi / 6})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=0 + zrot,
                        chi=0 - yrot,
                        phi=pi / 6,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                # Pair('100-->001', (cos(radians(4)), 0, sin(radians(4))),
                #     Position(mu=0, delta=60, nu=0, eta=0 + self.zrot, chi=4 - self.yrot,
                #         phi=30, unit='DEG'),),
                # Pair('010', (0, 1, 0),
                #     Position(mu=0, delta=60, nu=0, eta=30 + self.zrot, chi=0, phi=90, unit='DEG')),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6 + zrot,
                        chi=pi / 2 - yrot,
                        phi=pi / 6,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Position(
                        mu=0,
                        delta=radians(97.46959231642),
                        nu=0,
                        eta=radians(46.828815370173) + zrot,
                        chi=radians(86.69569481984) - yrot,
                        phi=pi / 6,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                # Pair('001-->100', (cos(radians(86)), 0, sin(radians(86))),
                #     Position(mu=0, delta=60, nu=0, eta=0 + self.zrot, chi=86 - self.yrot,
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
        self.hklcalc.constraints = Constraints({"mu": 0, "nu": 0, "phi": pi / 2})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6 + zrot,
                        chi=0,
                        phi=pi / 2,
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
        ("yrot"), [0, radians(2), radians(-2), pi / 4, -pi / 4, pi / 2, -pi / 2]
    )
    def test_pairs_various_zrot0_and_yrot(self, yrot, make_cases):
        case = make_cases(0, yrot)
        self.case_generator(case["010"])


class TestCubicVertical_MuEtaMode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints({"nu": 0, "mu": pi / 2, "eta": 0.0})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "011",
                    (0, 1, 1),
                    Position(
                        mu=pi / 2,
                        delta=pi / 2,
                        nu=0,
                        eta=0,
                        chi=0,
                        phi=pi / 2,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (sin(radians(4)), 0, cos(radians(4))),
                    Position(
                        mu=pi / 2,
                        delta=pi / 3,
                        nu=0,
                        eta=0,
                        chi=radians(56),
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=pi / 2,
                        delta=pi / 3,
                        nu=0,
                        eta=0,
                        chi=-pi / 6,
                        phi=pi / 2,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=pi / 2,
                        delta=pi / 3,
                        nu=0,
                        eta=0,
                        chi=pi / 3,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Position(
                        mu=pi / 2,
                        delta=radians(97.46959231642),
                        nu=0,
                        eta=0,
                        chi=radians(37.45112900750) - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->001",
                    (0, cos(radians(86)), sin(radians(86))),
                    Position(
                        mu=pi / 2,
                        delta=pi / 3,
                        nu=0,
                        eta=0,
                        chi=radians(56),
                        phi=pi / 2,
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
        self.hklcalc.constraints = Constraints({"psi": pi / 2, "mu": 0, "phi": 0})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=0,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->100",
                    (sin(radians(4)), cos(radians(4)), 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi * 2 / 3 - radians(4),
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
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi * 2 / 3,
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
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=pi / 2,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Position(
                        mu=0,
                        delta=radians(97.46959231642),
                        nu=0,
                        eta=radians(97.46959231642 / 2),
                        chi=radians(86.18592516571),
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001-->100",
                    (cos(radians(86)), 0, sin(radians(86))),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=radians(86),
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
        self.hklcalc.constraints = Constraints({"psi": 0, "eta": 0, "phi": 0})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=-pi / 2,
                        delta=pi / 3,
                        nu=0,
                        eta=0,
                        chi=pi / 6,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(radians(4)), 0, sin(radians(4))),
                    Position(
                        mu=-pi / 2,
                        delta=pi / 3,
                        nu=0,
                        eta=0,
                        chi=pi / 6 + radians(4),
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=pi * 2 / 3,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=pi,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=pi / 2,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 0.15),  # cover case where delta > 90 !
                    Position(
                        mu=-pi / 2,
                        delta=radians(10.34318),
                        nu=0,
                        eta=0,
                        chi=radians(61.48152),
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->001",
                    (0, cos(radians(4)), sin(radians(4))),
                    Position(
                        mu=radians(120 + 4),
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=pi,
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
        self.hklcalc.constraints = Constraints({"nu": 0, "bisect": True, "omega": 0})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "101",
                    (1, 0, 1),
                    Position(
                        mu=0,
                        delta=pi / 2,
                        nu=0,
                        eta=pi / 4,
                        chi=pi / 4,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "10m1",
                    (1, 0, -1),
                    Position(
                        mu=0,
                        delta=pi / 2,
                        nu=0,
                        eta=pi / 4,
                        chi=-pi / 4,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "011",
                    (0, 1, 1),
                    Position(
                        mu=0,
                        delta=pi / 2,
                        nu=0,
                        eta=pi / 4,
                        chi=pi / 4,
                        phi=pi / 2,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (sin(radians(4)), 0, cos(radians(4))),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=radians(86),
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=0,
                        phi=pi / 2,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=pi / 2,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.1 0 1.5",
                    (0.1, 0, 1.5),  # cover case where delta > 90 !
                    Position(
                        mu=0,
                        delta=radians(97.46959231642),
                        nu=0,
                        eta=radians(48.73480),
                        chi=radians(86.18593) - yrot,
                        phi=0 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->001",
                    (0, cos(radians(86)), sin(radians(86))),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=radians(86),
                        phi=pi / 2,
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
        self.hklcalc.constraints = Constraints({"nu": 0, "bisect": True, "mu": 0})


class TestCubicVertical_Bisect_qaz(TestCubicVertical_Bisect):
    def setup_method(self):
        TestCubicVertical_Bisect.setup_method(self)
        self.hklcalc.constraints = Constraints({"qaz": pi / 2, "bisect": True, "mu": 0})


class TestCubicHorizontal_Bisect(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints({"delta": 0, "bisect": True, "omega": 0})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "101",
                    (1, 0, 1),
                    Position(
                        mu=pi / 4,
                        delta=0,
                        nu=pi / 2,
                        eta=0,
                        chi=pi / 4,
                        phi=pi,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "10m1",
                    (1, 0, -1),
                    Position(
                        mu=pi / 4,
                        delta=0,
                        nu=pi / 2,
                        eta=0,
                        chi=radians(135),
                        phi=pi,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "011",
                    (0, 1, 1),
                    Position(
                        mu=pi / 4,
                        delta=0,
                        nu=pi / 2,
                        eta=0,
                        chi=pi / 4,
                        phi=pi * 3 / 2,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (sin(radians(4)), 0, cos(radians(4))),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=radians(4),
                        phi=pi,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=pi / 2,
                        phi=pi * 3 / 2,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
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
                    Position(
                        mu=radians(48.73480),
                        delta=0,
                        nu=radians(97.46959231642),
                        eta=0,
                        chi=radians(3.81407) - yrot,
                        phi=pi + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->001",
                    (0, cos(radians(86)), sin(radians(86))),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=radians(4),
                        phi=pi * 3 / 2,
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
        self.hklcalc.constraints = Constraints({"delta": 0, "bisect": True, "eta": 0})


class TestCubicHorizontal_Bisect_qaz(TestCubicHorizontal_Bisect):
    def setup_method(self):
        TestCubicHorizontal_Bisect.setup_method(self)
        self.hklcalc.constraints = Constraints({"qaz": 0, "bisect": True, "eta": 0})


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
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=pi / 2 + yrot,
                        phi=-pi + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(radians(4)), 0, sin(radians(4))),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=pi / 2 - radians(4) + yrot,
                        phi=-pi + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=pi / 2,
                        phi=-pi / 2 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),  # degenrate case mu||phi
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
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
                    (cos(radians(86)), 0, sin(radians(86))),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=-radians(4) - yrot,
                        phi=zrot,
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
        self.hklcalc.constraints = Constraints(constraint)
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
                radians(1),
            ],
            [radians(2), radians(-2)],
            [
                {"a_eq_b": True, "qaz": 0, "eta": 0},
                {"a_eq_b": True, "delta": 0, "eta": 0},
            ],
        ),
    )
    def test_hkl_to_angles_zrot_yrot(self, name, zrot, yrot, constraint, make_cases):
        self.hklcalc.constraints = Constraints(constraint)
        case = make_cases(zrot, yrot)
        if name == "010":
            with pytest.raises(DiffcalcException):
                self.case_generator(case[name])
        else:
            self.case_generator(case[name])


# class TestCubicHorizontal_qaz0_aeqb(_TestCubicHorizontal):
#    def setup_method(self):
#        _TestCubicHorizontal.setup_method(self)
#        self.hklcalc.constraints.asdict = {"a_eq_b": True, "qaz": 0, "eta": 0}
#
#
# class TestCubicHorizontal_delta0_aeqb(_TestCubicHorizontal):
#    def setup_method(self):
#        _TestCubicHorizontal.setup_method(self)
#        self.hklcalc.constraints.asdict = {"a_eq_b": True, "delta": 0, "eta": 0}


class TestCubic_FixedDetRefChiMode(_TestCubic):
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
                    Position(
                        mu=-pi / 2,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 2,
                        chi=pi / 2,
                        phi=-pi / 3,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(radians(4)), sin(radians(4)), 0),
                    Position(
                        mu=-pi / 2,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 2,
                        chi=pi / 2,
                        phi=-radians(56),
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=pi / 2,
                        delta=pi / 3,
                        nu=0,
                        eta=-pi / 2,
                        chi=pi / 2,
                        phi=-radians(150),
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=pi / 2,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001-->100",
                    (cos(radians(86)), sin(radians(86)), 0),
                    Position(
                        mu=pi / 2,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 2,
                        chi=pi / 2,
                        phi=radians(-34),
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
                {"a_eq_b": True, "qaz": pi / 2, "chi": pi / 2},
                {"a_eq_b": True, "nu": 0, "chi": pi / 2},
            ],
        ),
    )
    def test_pairs_zrot0_yrot0(self, name, constraint, make_cases):
        self.hklcalc.constraints = Constraints(constraint)
        case = make_cases(0, 0)
        if name == "001":
            with pytest.raises(DiffcalcException):
                self.case_generator(case[name])
        else:
            self.case_generator(case[name])


class TestCubic_FixedDeltaRefPhi0Mode(_TestCubic):
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
                    Position(
                        mu=pi * 2 / 3 - yrot,
                        delta=0,
                        nu=pi / 3,
                        eta=pi / 2 - zrot,
                        chi=pi,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(radians(4)), 0, sin(radians(4))),
                    Position(
                        mu=pi * 2 / 3 + radians(4) - yrot,
                        delta=0,
                        nu=pi / 3,
                        eta=pi / 2 - zrot,
                        chi=pi,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=pi * 2 / 3,
                        delta=0,
                        nu=pi / 3,
                        eta=-zrot,
                        chi=pi,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=pi / 6 - yrot,
                        delta=0,
                        nu=pi / 3,
                        eta=pi / 2 + zrot,
                        chi=0,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001-->100",
                    (cos(radians(86)), 0, sin(radians(86))),
                    Position(
                        mu=radians(30 - 4) - yrot,
                        delta=0,
                        nu=pi / 3,
                        eta=pi / 2 + zrot,
                        chi=0,
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
                radians(-1),
                radians(1),
            ],
            [radians(2)],
            [
                {"delta": 0, "psi": 0, "phi": 0},
                {"nu": pi / 3, "psi": 0, "phi": 0},
            ],
        ),
    )
    def test_hkl_to_angles_zrot_yrot(self, name, zrot, yrot, constraint, make_cases):
        self.hklcalc.constraints = Constraints(constraint)
        case = make_cases(zrot, yrot)
        self.case_generator(case[name])


class TestCubic_FixedDeltaEtaPhi0Mode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints({"eta": 0, "delta": 0, "phi": 0})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=-pi / 2 - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(radians(4)), 0, sin(radians(4))),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=radians(-90 + 4) - yrot,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=pi * 2 / 3,
                        delta=0,
                        nu=pi / 3,
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
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
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
                    (cos(radians(86)), 0, sin(radians(86))),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=radians(0 - 4) - yrot,
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
            [0, radians(2), radians(-2), pi / 4, -pi / 4, pi / 2, -pi / 2],
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
        self.hklcalc.constraints = Constraints({"eta": 0, "delta": 0, "phi": pi / 6})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=0,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=-pi / 2 - yrot,
                        phi=pi / 6,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=pi / 2,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=-pi / 2 - yrot,
                        phi=pi / 6,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=0 - yrot,
                        phi=pi / 6,
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
        self.hklcalc.constraints = Constraints({"eta": 0, "delta": 0, "chi": 0})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(zrot, yrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=pi * 2 / 3,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=0,
                        phi=-pi / 2 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (cos(radians(4)), 0, sin(radians(4))),
                    Position(
                        mu=radians(120 - 4),
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=0,
                        phi=-pi / 2 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=pi * 2 / 3,
                        delta=0,
                        nu=pi / 3,
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
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
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
                    (cos(radians(86)), 0, sin(radians(86))),
                    Position(
                        mu=radians(30 - 4),
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=0,
                        phi=pi / 2 + zrot,
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
            [0, radians(2), radians(-2)],
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
        self.hklcalc.constraints = Constraints({"eta": 0, "delta": 0, "chi": pi / 6})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=pi * 2 / 3,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=pi / 6,
                        phi=-pi / 2,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=pi * 2 / 3,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=pi / 6,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->001",
                    (-sin(pi / 6), 0, cos(pi / 6)),
                    Position(
                        mu=pi / 6,
                        delta=0,
                        nu=pi / 3,
                        eta=0,
                        chi=pi / 6,
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
        self.hklcalc.constraints = Constraints({"mu": 0, "nu": 0, "chi": pi / 2})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi * 2 / 3,
                        chi=pi / 2,
                        phi=-pi / 2 + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010-->100",
                    (sin(radians(4)), cos(radians(4)), 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi * 2 / 3,
                        chi=pi / 2,
                        phi=radians(-4) + zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi * 2 / 3,
                        chi=pi / 2,
                        phi=zrot,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "001",
                    (0, 0, 1),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi / 6,
                        chi=pi / 2,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),  # degenerate case phi||q
                Pair(
                    "100-->010",
                    (sin(radians(86)), cos(radians(86)), 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi * 2 / 3,
                        chi=pi / 2,
                        phi=radians(-90 + 4) + zrot,
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
            [0, radians(2), radians(-2)],
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
        self.hklcalc.constraints = Constraints({"mu": 0, "nu": 0, "chi": pi / 6})

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 1
            cases = (
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi * 2 / 3,
                        chi=pi / 6,
                        phi=-pi / 2,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi * 2 / 3,
                        chi=pi / 6,
                        phi=0,
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100-->010",
                    (sin(pi / 6), cos(pi / 6), 0),
                    Position(
                        mu=0,
                        delta=pi / 3,
                        nu=0,
                        eta=pi * 2 / 3,
                        chi=pi / 6,
                        phi=-pi / 6,
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

        self.hklcalc.constraints = Constraints({"a_eq_b": True, "mu": 0, "nu": 0})
        self.places = 2

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice("name", [3.8401, 5.43072])
        U = array(
            (
                (0.997161, -0.062217, 0.042420),
                (0.062542, 0.998022, -0.006371),
                (-0.041940, 0.009006, 0.999080),
            )
        )
        self.hklcalc.ubcalc.set_u(U)

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 1.24
            cases = (
                Pair(
                    "7_9_13",
                    (0.7, 0.9, 1.3),
                    Position(
                        mu=0,
                        delta=radians(27.352179),
                        nu=0,
                        eta=radians(13.676090),
                        chi=radians(37.774500),
                        phi=radians(53.965500),
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "100",
                    (1, 0, 0),
                    Position(
                        mu=0,
                        delta=radians(18.580230),
                        nu=0,
                        eta=radians(9.290115),
                        chi=radians(-2.403500),
                        phi=radians(3.589000),
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "010",
                    (0, 1, 0),
                    Position(
                        mu=0,
                        delta=radians(18.580230),
                        nu=0,
                        eta=radians(9.290115),
                        chi=radians(0.516000),
                        phi=radians(93.567000),
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "110",
                    (1, 1, 0),
                    Position(
                        mu=0,
                        delta=radians(26.394192),
                        nu=0,
                        eta=radians(13.197096),
                        chi=radians(-1.334500),
                        phi=radians(48.602000),
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
        self.hklcalc.constraints = Constraints({"phi": -pi / 2, "nu": 0, "mu": 0})
        self.wavelength = 12.39842 / 1.650

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice("xtal", [5.34, 13.2])
        self.hklcalc.ubcalc.set_u(I)

    def testHkl001(self):
        hkl = (0, 0, 1)
        pos = Position(
            mu=0,
            delta=radians(33.07329403295449),
            nu=0,
            eta=radians(16.536647016477247),
            chi=pi / 2,
            phi=-pi / 2,
        )
        self._check_angles_to_hkl("001", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("001", 999, 999, hkl, pos, self.wavelength, {})

    def testHkl100(self):
        hkl = (1, 0, 0)
        pos = Position(
            mu=0,
            delta=radians(89.42926563609406),
            nu=0,
            eta=radians(134.71463281804702),
            chi=0,
            phi=-pi / 2,
        )
        self._check_angles_to_hkl("100", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("100", 999, 999, hkl, pos, self.wavelength, {})

    def testHkl101(self):
        hkl = (1, 0, 1)
        pos = Position(
            mu=0,
            delta=radians(98.74666191021282),
            nu=0,
            eta=radians(117.347760720783),
            chi=pi / 2,
            phi=-pi / 2,
        )
        self._check_angles_to_hkl("101", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("101", 999, 999, hkl, pos, self.wavelength, {})


class TestThreeTwoCircleForDiamondI06andI10Horizontal(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)
        self.hklcalc.constraints = Constraints({"phi": -pi / 2, "delta": 0, "eta": 0})
        self.wavelength = 12.39842 / 1.650

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice("xtal", [5.34, 13.2])
        self.hklcalc.ubcalc.set_u(I)

    def testHkl001(self):
        hkl = (0, 0, 1)
        pos = Position(
            mu=radians(16.536647016477247),
            delta=0,
            nu=radians(33.07329403295449),
            eta=0,
            chi=0,
            phi=-pi / 2,
        )
        self._check_angles_to_hkl("001", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("001", 999, 999, hkl, pos, self.wavelength, {})

    @pytest.mark.xfail(raises=DiffcalcException)  # q || chi
    def testHkl100(self):
        hkl = (1, 0, 0)
        pos = Position(
            mu=radians(134.71463281804702),
            delta=0,
            nu=radians(89.42926563609406),
            eta=0,
            chi=0,
            phi=-pi / 2,
        )
        self._check_angles_to_hkl("100", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("100", 999, 999, hkl, pos, self.wavelength, {})

    def testHkl101(self):
        hkl = (1, 0, 1)
        pos = Position(
            mu=radians(117.347760720783),
            delta=0,
            nu=radians(98.74666191021282),
            eta=0,
            chi=0,
            phi=-pi / 2,
        )
        self._check_angles_to_hkl("101", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("101", 999, 999, hkl, pos, self.wavelength, {})


class TestThreeTwoCircleForDiamondI06andI10ChiDeltaEta(
    TestThreeTwoCircleForDiamondI06andI10Horizontal
):
    def setup_method(self):
        _BaseTest.setup_method(self)
        self.hklcalc.constraints = Constraints({"phi": -pi / 2, "chi": 0, "delta": 0})
        self.wavelength = 12.39842 / 1.650

    @pytest.mark.xfail(raises=DiffcalcException)  # q || eta
    def testHkl001(self):
        hkl = (0, 0, 1)
        pos = Position(
            mu=radians(16.536647016477247),
            delta=0,
            nu=radians(33.07329403295449),
            eta=0,
            chi=0,
            phi=-pi / 2,
        )
        self._check_angles_to_hkl("001", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("001", 999, 999, hkl, pos, self.wavelength, {})

    def testHkl100(self):
        hkl = (1, 0, 0)
        pos = Position(
            mu=radians(134.71463281804702),
            delta=0,
            nu=radians(89.42926563609406),
            eta=0,
            chi=0,
            phi=-pi / 2,
        )
        self._check_angles_to_hkl("100", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("100", 999, 999, hkl, pos, self.wavelength, {})


class TestFixedNazPsiEtaMode(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints({"naz": pi / 2, "psi": 0, "eta": 0})
        self.wavelength = 1

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_u(I)
        # Set some random reference vector orientation
        # that won't coincide with the scattering vector direction.
        # self.mock_ubcalc.n_phi = np.array([[0.087867277], [0.906307787], [0.413383038]])
        # self.mock_ubcalc.n_phi = np.array([[0.], [1.], [0.]])

    @pytest.mark.parametrize(
        ("hkl", "pos", "places"),
        [
            (
                (1, 0, 0),
                Position(
                    mu=-pi / 2,
                    delta=pi / 3,
                    nu=0,
                    eta=0,
                    chi=pi / 6,
                    phi=0,
                ),
                4,
            ),
            (
                (0, 1, 0),
                Position(
                    mu=pi / 2,
                    delta=pi / 3,
                    nu=0,
                    eta=0,
                    chi=radians(150),
                    phi=-pi / 2,
                ),
                4,
            ),
            (
                (1, 1, 0),
                Position(
                    mu=-pi / 2,
                    delta=pi / 2,
                    nu=0,
                    eta=0,
                    chi=pi / 4,
                    phi=pi / 4,
                ),
                4,
            ),
            (
                (1, 1, 1),
                Position(
                    mu=pi / 2,
                    delta=pi * 2 / 3,
                    nu=0,
                    eta=0,
                    chi=radians(84.7356),
                    phi=-radians(135),
                ),
                4,
            ),
            pytest.param(
                (0, 0, 1),
                Position(
                    mu=pi / 6,
                    delta=0,
                    nu=pi / 3,
                    eta=pi / 2,
                    chi=0,
                    phi=0,
                ),
                4,
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
        ],
    )
    def testHKL(self, hkl, pos, places):
        self.places = places
        self._check_angles_to_hkl("", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("", 999, 999, hkl, pos, self.wavelength, {})


class TestFixedChiPhiAeqBMode_DiamondI07SurfaceNormalHorizontal(_TestCubic):
    """
    The data here is taken from an experiment performed on Diamonds I07
    beamline, obtained using Vlieg's DIF software"""

    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "a_eq_b": True})
        self.wavelength = 1

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_u(I)
        # Set some random reference vector orientation
        # that won't coincide with the scattering vector direction.
        # self.mock_ubcalc.n_phi = np.array([[0.087867277], [0.906307787], [0.413383038]])
        # self.mock_ubcalc.n_phi = np.array([[0.], [1.], [0.]])

    @pytest.mark.parametrize(
        ("hkl", "pos", "places"),
        [
            pytest.param(
                (0, 0, 1),
                Position(
                    mu=pi / 6,
                    delta=0,
                    nu=pi / 3,
                    eta=pi / 2,
                    chi=0,
                    phi=0,
                ),
                4,
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (0, 1, 0),  # betaout=0
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=pi * 2 / 3,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (0, 1, 1),  # betaout=30
                Position(
                    mu=pi / 6,
                    delta=radians(54.7356),
                    nu=pi / 2,
                    eta=radians(125.2644),
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 0, 0),  # betaout=0
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=pi / 6,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 0, 1),  # betaout=30
                Position(
                    mu=pi / 6,
                    delta=radians(54.7356),
                    nu=pi / 2,
                    eta=radians(35.2644),
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 1, 0),  # betaout=0
                Position(
                    mu=0,
                    delta=pi / 2,
                    nu=0,
                    eta=pi / 2,
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 1, 0.0001),  # betaout=0
                Position(
                    mu=radians(0.0029),
                    delta=radians(89.9971),
                    nu=radians(90.0058),
                    eta=pi / 2,
                    chi=0,
                    phi=0,
                ),
                3,
            ),
            (
                (1, 1, 1),  # betaout=30
                Position(
                    mu=pi / 6,
                    delta=radians(54.7356),
                    nu=radians(150),
                    eta=radians(99.7356),
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (1.1, 0, 0),  # betaout=0
                Position(
                    mu=0,
                    delta=radians(66.7340),
                    nu=0,
                    eta=radians(33.3670),
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (0.9, 0, 0),  # betaout=0
                Position(
                    mu=0,
                    delta=radians(53.4874),
                    nu=0,
                    eta=radians(26.7437),
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (0.7, 0.8, 0.8),  # betaout=23.5782
                Position(
                    mu=radians(23.5782),
                    delta=radians(59.9980),
                    nu=radians(76.7037),
                    eta=radians(84.2591),
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (0.7, 0.8, 0.9),  # betaout=26.7437
                Position(
                    mu=radians(26.74368),
                    delta=radians(58.6754),
                    nu=radians(86.6919),
                    eta=radians(85.3391),
                    chi=0,
                    phi=0,
                ),
                4,
            ),
            (
                (0.7, 0.8, 1),  # betaout=30
                Position(
                    mu=pi / 6,
                    delta=radians(57.0626),
                    nu=radians(96.86590),
                    eta=radians(86.6739),
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


class TestFixedChiPhiAeqBModeSurfaceNormalVertical(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints(
            {"chi": pi / 2, "phi": 0, "a_eq_b": True}
        )
        self.wavelength = 1
        self.places = 4

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_u(I)

    @pytest.mark.parametrize(
        ("hkl", "pos", "places"),
        [
            pytest.param(
                (0, 0, 1),
                Position(
                    mu=pi / 6,
                    delta=0,
                    nu=pi / 3,
                    eta=pi / 2,
                    chi=0,
                    phi=0,
                ),
                4,
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (0, 1, 0),  # betaout=0
                Position(
                    mu=pi * 2 / 3,
                    delta=0,
                    nu=pi / 3,
                    eta=0,
                    chi=pi / 2,
                    phi=0,
                ),
                4,
            ),
            (
                (0, 1, 1),  # betaout=30
                Position(
                    mu=pi / 4,
                    delta=0,
                    nu=-pi / 2,
                    eta=radians(135),
                    chi=pi / 2,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 0, 0),  # betaout=0
                Position(
                    mu=-pi / 6,
                    delta=0,
                    nu=-pi / 3,
                    eta=0,
                    chi=pi / 2,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 0, 1),  # betaout=30
                Position(
                    mu=-pi / 6,
                    delta=radians(54.7356),
                    nu=-pi / 2,
                    eta=radians(35.2644),
                    chi=pi / 2,
                    phi=0,
                ),
                4,
            ),
            pytest.param(
                (1, -1, 0),  # betaout=0
                Position(
                    mu=-pi / 2,
                    delta=0,
                    nu=pi / 2,
                    eta=0,
                    chi=pi / 2,
                    phi=0,
                ),
                4,
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (1, 1, 0.0001),  # betaout=0
                Position(
                    mu=0,
                    delta=-radians(0.00286),
                    nu=-pi / 2,
                    eta=radians(179.997),
                    chi=pi / 2,
                    phi=0,
                ),
                4,
            ),
            (
                (1, 1, 1),  # betaout=30
                Position(
                    mu=-radians(171.5789),
                    delta=radians(20.9410),
                    nu=radians(122.3684),
                    eta=-radians(30.3612),
                    chi=pi / 2,
                    phi=0,
                ),
                4,
            ),
            (
                (1.1, 0, 0),  # betaout=0
                Position(
                    mu=-radians(146.6330),
                    delta=0,
                    nu=radians(66.7340),
                    eta=0,
                    chi=pi / 2,
                    phi=0,
                ),
                4,
            ),
            (
                (0.9, 0, 0),  # betaout=0
                Position(
                    mu=-radians(153.2563),
                    delta=0,
                    nu=radians(53.4874),
                    eta=0,
                    chi=pi / 2,
                    phi=0,
                ),
                4,
            ),
            (
                (0.7, 0.8, 0.8),  # betaout=23.5782
                Position(
                    mu=radians(167.7652),
                    delta=radians(23.7336),
                    nu=radians(82.7832),
                    eta=-radians(24.1606),
                    chi=pi / 2,
                    phi=0,
                ),
                4,
            ),
            (
                (0.7, 0.8, 0.9),  # betaout=26.7437
                Position(
                    mu=radians(169.0428),
                    delta=radians(25.6713),
                    nu=radians(88.0926),
                    eta=-radians(27.2811),
                    chi=pi / 2,
                    phi=0,
                ),
                4,
            ),
            (
                (0.7, 0.8, 1),  # betaout=30
                Position(
                    mu=radians(170.5280),
                    delta=radians(27.1595),
                    nu=radians(94.1895),
                    eta=-radians(30.4583),
                    chi=pi / 2,
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


class TestFixedChiPhiPsiModeSurfaceNormalVerticalI16(_TestCubic):
    # testing with Chris N. for pre christmas 2012 i16 experiment

    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints({"chi": pi / 2, "psi": pi / 2, "phi": 0})
        self.wavelength = 1
        self.places = 4

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_u(I)

    @pytest.mark.parametrize(
        ("hkl", "pos"),
        [
            pytest.param(
                (1, 1, 0),  # Any qaz can be set in [0, 90] with eta and delta
                Position(
                    mu=pi / 2,
                    delta=0,
                    nu=pi / 2,
                    eta=0,
                    chi=-pi / 2,
                    phi=0,
                ),
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (1, 1, 0.001),
                Position(
                    mu=-radians(89.9714),
                    delta=radians(90.0430),
                    nu=-radians(89.9618),
                    eta=radians(90.0143),
                    chi=pi / 2,
                    phi=0,
                ),
            ),
            (
                (1, 1, 0.1),
                Position(
                    mu=-radians(87.1331),
                    delta=radians(85.6995),
                    nu=radians(93.8232),
                    eta=radians(91.4339),
                    chi=pi / 2,
                    phi=0,
                ),
            ),
            (
                (1, 1, 0.5),
                Position(
                    mu=-radians(75.3995),
                    delta=radians(68.0801),
                    nu=radians(109.5630),
                    eta=radians(97.3603),
                    chi=pi / 2,
                    phi=0,
                ),
            ),
            (
                (1, 1, 1),
                Position(
                    mu=-radians(58.6003),
                    delta=radians(42.7342),
                    nu=radians(132.9005),
                    eta=radians(106.3250),
                    chi=pi / 2,
                    phi=0,
                ),
            ),
        ],
    )
    def testHKL(self, hkl, pos):
        self._check_angles_to_hkl("", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("", 999, 999, hkl, pos, self.wavelength, {})


class TestConstrain3Sample_ChiPhiEta(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints(
            {"chi": pi / 2, "phi": 0, "a_eq_b": True}
        )
        self.wavelength = 1
        self.places = 4

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_u(I)
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
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (0, 0, 1),
            Position(
                mu=pi / 6,
                delta=0,
                nu=pi / 3,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_all0_010(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (0, 1, 0),
            Position(
                mu=pi * 2 / 3,
                delta=0,
                nu=pi / 3,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_all0_011(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (0, 1, 1),
            Position(
                mu=pi / 2,
                delta=0,
                nu=pi / 2,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_phi30_100(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": pi / 6, "eta": 0})
        self._check(
            (1, 0, 0),
            Position(
                mu=0,
                delta=pi / 3,
                nu=0,
                eta=0,
                chi=0,
                phi=pi / 6,
            ),
        )

    def testHkl_eta30_100(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": pi / 6})
        self._check(
            (1, 0, 0),
            Position(
                mu=0,
                delta=pi / 3,
                nu=0,
                eta=pi / 6,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_phi90_110(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": pi / 2, "eta": 0})
        self._check(
            (1, 1, 0),
            Position(
                mu=0,
                delta=pi / 2,
                nu=0,
                eta=0,
                chi=0,
                phi=pi / 2,
            ),
        )

    def testHkl_eta90_110(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": pi / 2})
        self._check(
            (1, 1, 0),
            Position(
                mu=0,
                delta=pi / 2,
                nu=0,
                eta=pi / 2,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_all0_1(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (0.01, 0.01, 0.1),
            Position(
                mu=radians(8.6194),
                delta=radians(0.5730),
                nu=radians(5.7607),
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_all0_2(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (0, 0, 0.1),
            Position(
                mu=radians(2.8660),
                delta=0,
                nu=radians(5.7320),
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_all0_3(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (0.1, 0, 0.01),
            Position(
                mu=radians(30.3314),
                delta=radians(5.7392),
                nu=radians(0.4970),
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_show_all_solutionsall0_3(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (0.1, 0, 0.01),
            Position(
                mu=radians(30.3314),
                delta=radians(5.7392),
                nu=radians(0.4970),
                eta=0,
                chi=0,
                phi=0,
            ),
        )
        # print self.hklcalc.hkl_to_all_angles(.1, 0, .01, 1)

    def testHkl_all0_010to001(self):
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (0, cos(radians(4)), sin(radians(4))),
            Position(
                mu=radians(120 - 4),
                delta=0,
                nu=pi / 3,
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_1(self):
        self.wavelength = 0.1
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (0, 0, 1),
            Position(
                mu=radians(2.8660),
                delta=0,
                nu=radians(5.7320),
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_2(self):
        self.wavelength = 0.1
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (0, 0, 1),
            Position(
                mu=radians(2.8660),
                delta=0,
                nu=radians(5.7320),
                eta=0,
                chi=0,
                phi=0,
            ),
        )

    def testHkl_3(self):
        self.wavelength = 0.1
        self.hklcalc.constraints = Constraints({"chi": 0, "phi": 0, "eta": 0})
        self._check(
            (1, 0, 0.1),
            Position(
                mu=radians(30.3314),
                delta=radians(5.7392),
                nu=radians(0.4970),
                eta=0,
                chi=0,
                phi=0,
            ),
        )


class TestConstrain3Sample_MuChiPhi(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints(
            {"chi": pi / 2, "phi": 0, "a_eq_b": True}
        )
        self.wavelength = 1
        self.places = 4

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_u(I)
        # Set some random reference vector orientation
        # that won't coincide with the scattering vector direction.
        # self.mock_ubcalc.n_phi = np.array([[0.087867277], [0.906307787], [0.413383038]])

    @pytest.mark.parametrize(
        ("hkl", "pos", "constraint"),
        [
            (
                (0, 0, 1),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=pi / 6,
                    chi=pi / 2,
                    phi=0,
                ),
                {"chi": pi / 2, "phi": 0, "mu": 0},
            ),
            (
                (0, 1, 0),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=pi * 2 / 3,
                    chi=pi / 2,
                    phi=0,
                ),
                {"chi": pi / 2, "phi": 0, "mu": 0},
            ),
            (
                (0, 1, 1),
                Position(
                    mu=0,
                    delta=pi / 2,
                    nu=0,
                    eta=pi / 2,
                    chi=pi / 2,
                    phi=0,
                ),
                {"chi": pi / 2, "phi": 0, "mu": 0},
            ),
            (
                (1, 0, 0),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=-pi / 3,
                    chi=pi / 2,
                    phi=pi / 2,
                ),
                {"chi": pi / 2, "phi": pi / 2, "mu": 0},
            ),
            pytest.param(
                (-1, 1, 0),
                Position(
                    mu=pi / 2,
                    delta=pi / 2,
                    nu=0,
                    eta=pi / 2,
                    chi=pi / 2,
                    phi=0,
                ),
                {"chi": pi / 2, "phi": 0, "mu": pi / 2},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (1, 0, 1),
                Position(
                    mu=0,
                    delta=pi / 2,
                    nu=0,
                    eta=0,
                    chi=pi / 2,
                    phi=pi / 2,
                ),
                {"chi": pi / 2, "phi": pi / 2, "mu": 0},
            ),
            (
                (sin(radians(4)), cos(radians(4)), 0),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=radians(120 - 4),
                    chi=0,
                    phi=0,
                ),
                {"chi": 0, "phi": 0, "mu": 0},
            ),
        ],
    )
    def testHkl(self, hkl, pos, constraint):
        self.hklcalc.constraints = Constraints(constraint)
        self._check_angles_to_hkl("", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("", 999, 999, hkl, pos, self.wavelength, {})


class TestConstrain3Sample_MuEtaChi(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.wavelength = 1
        self.places = 4

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_u(I)

    @pytest.mark.parametrize(
        ("hkl", "pos", "constraint"),
        [
            pytest.param(
                (0, 0, 1),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=pi / 6,
                    chi=pi / 2,
                    phi=0,
                ),
                {"eta": pi / 6, "chi": pi / 2, "mu": 0},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (0, 1, 0),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=pi * 2 / 3,
                    chi=pi / 2,
                    phi=0,
                ),
                {"chi": pi / 2, "eta": pi * 2 / 3, "mu": 0},
            ),
            pytest.param(
                (0, 1, 1),
                Position(
                    mu=0,
                    delta=pi / 2,
                    nu=0,
                    eta=pi / 2,
                    chi=pi / 2,
                    phi=0,
                ),
                {"chi": pi / 2, "eta": pi / 2, "mu": 0},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (1, 0, 0),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=-pi / 3,
                    chi=pi / 2,
                    phi=pi / 2,
                ),
                {"chi": pi / 2, "eta": -pi / 3, "mu": 0},
            ),
            (
                (-1, 1, 0),
                Position(
                    mu=pi / 2,
                    delta=pi / 2,
                    nu=0,
                    eta=pi / 2,
                    chi=pi / 2,
                    phi=0,
                ),
                {"chi": pi / 2, "eta": pi / 2, "mu": pi / 2},
            ),
            (
                (1, 0, 1),
                Position(
                    mu=0,
                    delta=pi / 2,
                    nu=0,
                    eta=0,
                    chi=pi / 2,
                    phi=pi / 2,
                ),
                {"chi": pi / 2, "eta": 0, "mu": 0},
            ),
            (
                (sin(radians(4)), cos(radians(4)), 0),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=0,
                    chi=0,
                    phi=radians(120 - 4),
                ),
                {"chi": 0, "eta": 0, "mu": 0},
            ),
        ],
    )
    def testHKL(self, hkl, pos, constraint):
        self.hklcalc.constraints = Constraints(constraint)
        self._check_angles_to_hkl("", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("", 999, 999, hkl, pos, self.wavelength, {})


class TestConstrain3Sample_MuEtaPhi(_TestCubic):
    def setup_method(self):
        _TestCubic.setup_method(self)
        self.hklcalc.constraints = Constraints(
            {"chi": pi / 2, "phi": 0, "a_eq_b": True}
        )
        self.wavelength = 1
        self.places = 4

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_u(I)

    @pytest.mark.parametrize(
        ("hkl", "pos", "constraint"),
        [
            (
                (0, 0, 1),
                Position(
                    mu=pi / 6,
                    delta=0,
                    nu=pi / 3,
                    eta=0,
                    chi=0,
                    phi=0,
                ),
                {"eta": 0, "phi": 0, "mu": pi / 6},
            ),
            pytest.param(
                (0, 1, 0),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=pi * 2 / 3,
                    chi=pi / 2,
                    phi=0,
                ),
                {"eta": pi * 2 / 3, "phi": 0, "mu": 0},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (0, 1, 1),
                Position(
                    mu=0,
                    delta=pi / 2,
                    nu=0,
                    eta=pi / 2,
                    chi=pi / 2,
                    phi=0,
                ),
                {"eta": pi / 2, "phi": 0, "mu": 0},
            ),
            pytest.param(
                (1, 0, 0),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=-pi / 3,
                    chi=pi / 2,
                    phi=pi / 2,
                ),
                {"eta": -pi / 3, "phi": pi / 2, "mu": 0},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (-1, 1, 0),
                Position(
                    mu=pi / 2,
                    delta=pi / 2,
                    nu=0,
                    eta=pi / 2,
                    chi=pi / 2,
                    phi=0,
                ),
                {"eta": pi / 2, "phi": 0, "mu": pi / 2},
            ),
            pytest.param(
                (1, 0, 1),
                Position(
                    mu=0,
                    delta=pi / 2,
                    nu=0,
                    eta=0,
                    chi=pi / 2,
                    phi=pi / 2,
                ),
                {"eta": 0, "phi": pi / 2, "mu": 0},
                marks=pytest.mark.xfail(raises=DiffcalcException),
            ),
            (
                (sin(radians(4)), 0, cos(radians(4))),
                Position(
                    mu=0,
                    delta=pi / 3,
                    nu=0,
                    eta=pi / 6,
                    chi=pi / 2 - radians(4),
                    phi=0,
                ),
                {"eta": pi / 6, "phi": 0, "mu": 0},
            ),
        ],
    )
    def testHKL(self, hkl, pos, constraint):
        self.hklcalc.constraints = Constraints(constraint)
        self._check_angles_to_hkl("", 999, 999, hkl, pos, self.wavelength, {})
        self._check_hkl_to_angles("", 999, 999, hkl, pos, self.wavelength, {})


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
        self.hklcalc.ubcalc.set_lattice("I16_test", [4.785, 12.991], "Hexagonal")

        U = array(
            [
                [-9.65616334e-01, -2.59922060e-01, 5.06142415e-03],
                [2.59918682e-01, -9.65629598e-01, -1.32559487e-03],
                [5.23201232e-03, 3.55426382e-05, 9.99986312e-01],
            ]
        )
        self.hklcalc.ubcalc.set_u(U)

    def _check(self, hkl, pos, virtual_expected={}, skip_test_pair_verification=False):
        if not skip_test_pair_verification:
            self._check_angles_to_hkl(
                "", 999, 999, hkl, pos, self.wavelength, virtual_expected
            )
        self._check_hkl_to_angles(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )

    def test_hkl_bisecting_works_okay_on_i16(self):
        self.hklcalc.constraints = Constraints({"delta": 0, "a_eq_b": True, "eta": 0})
        self._check(
            [-1.1812112493619709, -0.71251524866987204, 5.1997083010199221],
            Position(
                mu=radians(26),
                delta=0,
                nu=radians(52),
                eta=0,
                chi=radians(45.2453),
                phi=radians(186.6933 - 360),
            ),
        )

    def test_hkl_psi90_works_okay_on_i16(self):
        # This is failing here but on the live one. Suggesting some extreme sensitivity?
        self.hklcalc.constraints = Constraints({"delta": 0, "psi": -pi / 2, "eta": 0})
        self._check(
            [-1.1812112493619709, -0.71251524866987204, 5.1997083010199221],
            Position(
                mu=radians(26),
                delta=0,
                nu=radians(52),
                eta=0,
                chi=radians(45.2453),
                phi=radians(186.6933 - 360),
            ),
        )

    def test_hkl_alpha_17_9776_used_to_fail(self):
        # This is failing here but on the live one. Suggesting some extreme sensitivity?
        self.hklcalc.constraints = Constraints(
            {"delta": 0, "alpha": radians(17.9776), "eta": 0}
        )
        self._check(
            [-1.1812112493619709, -0.71251524866987204, 5.1997083010199221],
            Position(
                mu=radians(26),
                delta=0,
                nu=radians(52),
                eta=0,
                chi=radians(45.2453),
                phi=radians(186.6933 - 360),
            ),
        )

    def test_hkl_alpha_17_9776_failing_after_bigger_small(self):
        # This is failing here but on the live one. Suggesting some extreme sensitivity?
        self.hklcalc.constraints = Constraints(
            {"delta": 0, "alpha": radians(17.8776), "eta": 0}
        )
        self._check(
            [-1.1812112493619709, -0.71251524866987204, 5.1997083010199221],
            Position(
                mu=radians(25.85),
                delta=0,
                nu=radians(52),
                eta=0,
                chi=radians(45.2453),
                phi=radians(-173.518),
            ),
        )


# skip_test_pair_verification


class TestAnglesToHkl_I16Examples:
    def setup_method(self):
        self.hklcalc = HklCalculation()

        U = array(
            (
                (0.9996954135095477, -0.01745240643728364, -0.017449748351250637),
                (0.01744974835125045, 0.9998476951563913, -0.0003045864904520898),
                (0.017452406437283505, -1.1135499981271473e-16, 0.9998476951563912),
            )
        )
        self.WL1 = 1  # Angstrom
        self.hklcalc.ubcalc.set_lattice("Cubic", [1.0])
        self.hklcalc.ubcalc.set_u(U)

    def test_anglesToHkl_mu_0_gam_0(self):
        pos = PosFromI16sEuler(radians(1), radians(1), pi / 6, 0, pi / 3, 0)
        arrayeq_(self.hklcalc.get_hkl(pos, self.WL1), [1, 0, 0])

    def test_anglesToHkl_mu_0_gam_10(self):
        pos = PosFromI16sEuler(radians(1), radians(1), pi / 6, 0, pi / 3, radians(10))
        arrayeq_(
            self.hklcalc.get_hkl(pos, self.WL1),
            [1.00379806, -0.006578435, 0.08682408],
        )

    def test_anglesToHkl_mu_10_gam_0(self):
        pos = PosFromI16sEuler(radians(1), radians(1), pi / 6, radians(10), pi / 3, 0)
        arrayeq_(
            self.hklcalc.get_hkl(pos, self.WL1),
            [0.99620193, 0.0065784359, 0.08682408],
        )

    def test_anglesToHkl_arbitrary(self):
        pos = PosFromI16sEuler(
            radians(1.9),
            radians(2.9),
            radians(30.9),
            radians(0.9),
            radians(60.9),
            radians(2.9),
        )
        arrayeq_(
            self.hklcalc.get_hkl(pos, self.WL1),
            [1.01174189, 0.02368622, 0.06627361],
        )


class TestAnglesToHkl_I16Numerical(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)

        # self.UB = array(((1.11143, 0, 0), (0, 1.11143, 0), (0, 0, 1.11143)))

        self.hklcalc.constraints = Constraints({"mu": 0, "nu": 0, "phi": 0})
        self.wavelength = 1.0
        self.places = 6

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice("xtal", [5.653244295348863])
        self.hklcalc.ubcalc.set_u(I)
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
            PosFromI16sEuler(
                0, radians(0.000029), radians(10.188639), 0, radians(20.377277), 0
            ),
        )
        self._check(
            "I16_numeric",
            [2, 0.000001, 0],
            PosFromI16sEuler(0, 0, radians(10.188667), 0, radians(20.377277), 0),
        )


class TestAnglesToHkl_I16GaAsExample(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)

        self.hklcalc.constraints = Constraints(
            {
                "qaz": pi / 2,
                "alpha": radians(11.0),
                "mu": 0.0,
            }
        )
        self.wavelength = 1.239842
        self.places = 3

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice("xtal", [5.65315])
        U = array(
            [
                [-0.71021455, 0.70390373, 0.01071626],
                [-0.39940627, -0.41542895, 0.81724747],
                [0.57971538, 0.5761409, 0.57618724],
            ]
        )
        self.hklcalc.ubcalc.set_u(U)

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
            PosFromI16sEuler(
                radians(10.8224),
                radians(89.8419),
                radians(11.0000),
                0.0,
                radians(21.8980),
                0.0,
            ),
        )
        self._check(
            [0.0, 0.0, 2.0],
            PosFromI16sEuler(
                radians(81.2389),
                radians(35.4478),
                radians(19.2083),
                0.0,
                radians(25.3375),
                0.0,
            ),
        )


class Test_I21ExamplesUB(_BaseTest):
    """NOTE: copied from test.diffcalc.scenarios.session3"""

    def setup_method(self):
        _BaseTest.setup_method(self)

        # self.hklcalc.constraints = Constraints()
        # self.hklcalc = HklCalculation(self.ubcalc, self.hklcalc.constraints)

        # B = array(((1.66222, 0.0, 0.0), (0.0, 1.66222, 0.0), (0.0, 0.0, 0.31260)))

        self.hklcalc.constraints = Constraints({"psi": radians(10), "mu": 0, "nu": 0})
        self.places = 3

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice("xtal", [3.78, 20.10])
        U = array(((1.0, 0.0, 0.0), (0.0, 0.18482, -0.98277), (0.0, 0.98277, 0.18482)))
        self.hklcalc.ubcalc.set_u(U)
        self.hklcalc.ubcalc.n_phi = (0, 0, 1)

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 12.39842 / 0.650
            cases = (
                Pair(
                    "0_0.2_0.25",
                    (0.0, 0.2, 0.25),
                    Position(
                        mu=0,
                        delta=radians(62.44607),
                        nu=0,
                        eta=radians(28.68407),
                        chi=radians(90.0 - 0.44753),
                        phi=-radians(9.99008),
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "0.25_0.2_0.1",
                    (0.25, 0.2, 0.1),
                    Position(
                        mu=0,
                        delta=radians(108.03033),
                        nu=0,
                        eta=radians(3.03132),
                        chi=radians(90 - 7.80099),
                        phi=radians(87.95201),
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


class Test_FixedAlphaMuChiSurfaceNormalHorizontal(_BaseTest):
    """NOTE: copied from test.diffcalc.scenarios.session3"""

    def setup_method(self):
        _BaseTest.setup_method(self)

        self.hklcalc.constraints = Constraints(
            {"alpha": radians(12.0), "mu": 0, "chi": pi / 2}
        )
        self.places = 4

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice("GaAs", [5.65325])
        U = array(
            (
                (-0.71022, 0.70390, 0.01071),
                (-0.39941, -0.41543, 0.81725),
                (0.57971, 0.57615, 0.57618),
            )
        )

        self.hklcalc.ubcalc.set_u(U)
        self.hklcalc.ubcalc.n_hkl = (0, 0, 1)

    @pytest.fixture(scope="class")
    def make_cases(self):
        def __make_cases_fixture(yrot, zrot):
            wavelength = 1.0
            cases = (
                Pair(
                    "2_2_2",
                    (2.0, 2.0, 2.0),
                    Position(
                        mu=0,
                        delta=radians(35.6825),
                        nu=radians(0.0657),
                        eta=radians(17.9822),
                        chi=pi / 2,
                        phi=radians(-92.9648),
                    ),
                    zrot,
                    yrot,
                    wavelength,
                ),
                Pair(
                    "2_2_0",
                    (2.0, 2.0, 0.0),
                    Position(
                        mu=0,
                        delta=radians(22.9143),
                        nu=radians(18.2336),
                        eta=radians(18.7764),
                        chi=pi / 2,
                        phi=radians(90.9119),
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
            "2_2_2",
            "2_2_0",
        ],
    )
    def test_hkl_to_angles_given_UB(self, name, make_cases):
        case = make_cases(0, 0)
        self.case_generator(case[name])


class TestConstrainNazAlphaEta(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)

        self.hklcalc.constraints = Constraints(
            {
                "naz": radians(3.0),
                "alpha": radians(2.0),
                "eta": radians(1.0),
            }
        )
        self.wavelength = 1
        self.places = 4

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice("test", [4.913, 5.405])
        self.hklcalc.ubcalc.add_reflection(
            hkl=(0, 0, 1),
            position=Position(radians(7.31), 0, radians(10.62), 0, 0, 0),
            energy=12.39842,
            tag="refl1",
        )
        self.hklcalc.ubcalc.add_orientation(hkl=(0, 1, 0), xyz=(0, 1, 0), tag="plane")
        self.hklcalc.ubcalc.calc_ub("refl1", "plane")

        self.hklcalc.ubcalc.n_hkl = (1.0, 0.0, 0.0)

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
            Position(
                radians(90.8684),
                radians(-15.0274),
                radians(12.8921),
                radians(1.0),
                radians(-29.5499),
                radians(-87.7027),
            ),
        )
        self._check(
            [1.0, 2.0, 3.0],
            Position(
                radians(90.8265),
                radians(-39.0497),
                radians(17.0693),
                radians(1.0),
                radians(-30.4506),
                radians(-87.6817),
            ),
        )

    @pytest.mark.xfail(raises=DiffcalcException)
    def test_hkl_to_angles_no_solution(self):
        self._check(
            [1.0, 0.0, 0.0],
            Position(radians(5.8412), 0.0, radians(11.6823), 0.0, -pi / 2, 0),
        )
