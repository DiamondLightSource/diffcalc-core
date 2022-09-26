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

# TODO: class largely copied from test_calc

from collections import namedtuple
from math import pi, radians

import pytest
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.ub.crystal import Crystal
from diffcalc.util import DiffcalcException, I
from numpy import array

from tests.diffcalc.hkl.test_calc import _BaseTest
from tests.tools import matrixeq_


class TestSurfaceNormalVerticalCubic(_BaseTest):
    def setup_method(self):

        _BaseTest.setup_method(self)
        self.hklcalc.constraints = Constraints(
            {"a_eq_b": True, "mu": -pi / 2, "eta": 0}
        )
        self.wavelength = 1

    def _configure_ub(self):
        self.hklcalc.ubcalc.n_phi = (1, 0, 0)
        self.hklcalc.ubcalc.set_lattice("xtal", [1.0])
        self.hklcalc.ubcalc.set_u(I)

    def _check(self, hkl, pos, virtual_expected={}):
        if pos is not None:
            self._check_angles_to_hkl(
                "", 999, 999, hkl, pos, self.wavelength, virtual_expected
            )
        self._check_hkl_to_angles(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )

    def testHkl001(self):
        pos = Position(
            mu=-pi / 2,
            delta=pi / 3,
            nu=0,
            eta=0,
            chi=pi * 2 / 3,  # 90+ 30 = 4pi/6 = 2pi/3
            phi=pi / 2,
        )
        self._check((0, 0, 1), pos, {"betain": pi / 6, "betaout": pi / 6})

    def testHkl011(self):
        pos = Position(
            mu=-pi / 2,
            delta=pi / 2,
            nu=0,
            eta=0,
            chi=pi / 2,
            phi=pi / 2,
        )
        self._check((0, 1, 1), pos, {"betain": 0, "betaout": pi / 2})

    def testHkl010(self):
        pos = Position(
            mu=-pi / 2,
            delta=pi / 3,
            nu=0,
            eta=0,
            chi=pi / 6,
            phi=pi / 2,
        )
        self._check((0, 1, 0), pos, {"betain": -pi / 3, "betaout": pi / 3})

    @pytest.mark.xfail(raises=DiffcalcException)
    def testHkl100(self):
        self._check((1, 0, 0), None, {"alpha": pi / 6, "beta": pi / 6})

    def testHkl110(self):
        pos = Position(
            mu=-pi / 2,
            delta=pi / 2,
            nu=0,
            eta=0,
            chi=pi / 4,
            phi=pi / 4,
        )
        self._check((1, 1, 0), pos, {"alpha": pi / 6, "beta": pi / 6})


# Primary and secondary reflections found with the help of DDIF on Diamond's
# i07 on Jan 27 2010
WillPos = namedtuple("WillPos", ["delta", "gamma", "omegah", "phi"])

HKL0 = 2, 19, 32
REF0 = WillPos(
    delta=radians(21.975), gamma=radians(4.419), omegah=radians(2), phi=radians(326.2)
)

HKL1 = 0, 7, 22
REF1 = WillPos(
    delta=radians(11.292), gamma=radians(2.844), omegah=radians(2), phi=radians(124.1)
)

WAVELENGTH = 0.6358
ENERGY = 12.39842 / WAVELENGTH


# This is the version that Diffcalc comes up with ( see following test)
U_DIFFCALC = array(
    [
        [-0.7178876, 0.6643924, -0.2078944],
        [-0.6559596, -0.5455572, 0.5216170],
        [0.2331402, 0.5108327, 0.8274634],
    ]
)


# class WillmottHorizontalGeometry(VliegGeometry):
#
#    def __init__(self):
#        VliegGeometry.__init__(self,
#                    name='willmott_horizontal',
#                    supported_mode_groups=[],
#                    fixed_parameters={},
#                    gamma_location='base'
#                    )
#
#    def physical_angles_to_internal_position(self, physicalAngles):
#        assert (len(physicalAngles) == 4), "Wrong length of input list"
#        return WillPos(*physicalAngles)
#
#    def internal_position_to_physical_angles(self, internalPosition):
#        return internalPosition.totuple()


def willmott_to_you_fixed_mu_eta(pos):
    pos = Position(
        mu=-pi / 2,
        delta=pos.delta,
        nu=pos.gamma,
        eta=0,
        chi=pi / 2 + pos.omegah,
        phi=-pi / 2 - pos.phi,
    )
    return pos


class TestUBCalculationWithWillmotStrategy_Si_5_5_12_FixedMuEta:
    def testAgainstResultsFromJan_27_2010(self):
        self.ubcalc = UBCalculation()
        self.ubcalc.set_lattice(
            "Si_5_5_12", [7.68, 53.48, 75.63, pi / 2, pi / 2, pi / 2]
        )
        self.ubcalc.add_reflection(
            HKL0,
            willmott_to_you_fixed_mu_eta(REF0),
            ENERGY,
            "ref0",
        )
        self.ubcalc.add_reflection(
            HKL1,
            willmott_to_you_fixed_mu_eta(REF1),
            ENERGY,
            "ref1",
        )
        self.ubcalc.calc_ub()
        print("U: ", self.ubcalc.U)
        print("UB: ", self.ubcalc.UB)
        matrixeq_(self.ubcalc.U, U_DIFFCALC)


class TestFixedMuEta(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)
        self.hklcalc.constraints = Constraints(
            {"alpha": radians(2), "mu": -pi / 2, "eta": 0}
        )
        self.wavelength = 0.6358

    def _configure_constraints(self):
        self.hklcalc.constraints.asdict = {"alpha": 2, "mu": -90, "eta": 0}

    def _convert_willmott_pos(self, willmott_pos):
        return willmott_to_you_fixed_mu_eta(willmott_pos)

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice(
            "xtal", [7.68, 53.48, 75.63, pi / 2, pi / 2, pi / 2]
        )
        self.hklcalc.ubcalc.set_u(U_DIFFCALC)

    def _check(self, hkl, pos, virtual_expected={}):
        self._check_angles_to_hkl(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )
        self._check_hkl_to_angles(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )

    def testHkl_2_19_32_found_orientation_setting(self):
        """Check that the or0 reflection maps back to the assumed hkl"""
        self.places = 2
        self._check_angles_to_hkl(
            "",
            999,
            999,
            HKL0,
            willmott_to_you_fixed_mu_eta(REF0),
            self.wavelength,
            {"alpha": radians(2)},
        )

    def testHkl_0_7_22_found_orientation_setting(self):
        """Check that the or1 reflection maps back to the assumed hkl"""
        self.places = 0
        self._check_angles_to_hkl(
            "",
            999,
            999,
            HKL1,
            willmott_to_you_fixed_mu_eta(REF1),
            self.wavelength,
            {"alpha": radians(2)},
        )

    def testHkl_2_19_32_calculated_from_DDIF(self):
        self.places = 3
        willpos = WillPos(
            delta=radians(21.974),
            gamma=radians(4.419),
            omegah=radians(2),
            phi=radians(-33.803),
        )
        self._check(
            (2, 19, 32), willmott_to_you_fixed_mu_eta(willpos), {"alpha": radians(2)}
        )

    def testHkl_0_7_22_calculated_from_DDIF(self):
        self.places = 3
        willpos = WillPos(
            delta=radians(11.241801854649),
            gamma=radians(-3.038407637123),
            omegah=radians(2),
            phi=radians(-86.56344250267),
        )
        self._check(
            (0, 7, 22), willmott_to_you_fixed_mu_eta(willpos), {"alpha": radians(2)}
        )

    def testHkl_2_m5_12_calculated_from_DDIF(self):
        self.places = 3
        willpos = WillPos(
            delta=radians(5.224),
            gamma=radians(10.415),
            omegah=radians(2),
            phi=radians(-1.972),
        )
        self._check(
            (2, -5, 12), willmott_to_you_fixed_mu_eta(willpos), {"alpha": radians(2)}
        )

    def testHkl_2_19_32_calculated_predicted_with_diffcalc_and_found(self):
        willpos = WillPos(
            delta=radians(21.974032376045),
            gamma=radians(4.418955754003),
            omegah=radians(2),
            phi=radians(-33.80254),
        )
        self._check(
            (2, 19, 32), willmott_to_you_fixed_mu_eta(willpos), {"alpha": radians(2)}
        )

    def testHkl_0_7_22_calculated_predicted_with_diffcalc_and_found(self):
        willpos = WillPos(
            delta=radians(11.241801854649),
            gamma=radians(-3.038407637123),
            omegah=radians(2),
            phi=radians(-86.563442502670),
        )
        self._check(
            (0, 7, 22), willmott_to_you_fixed_mu_eta(willpos), {"alpha": radians(2)}
        )

    def testHkl_2_m5_12_calculated_predicted_with_diffcalc_and_found(self):
        willpos = WillPos(
            delta=radians(5.223972025344),
            gamma=radians(10.415435905622),
            omegah=radians(2),
            phi=radians(-90 + 88.02751),
        )
        self._check(
            (2, -5, 12), willmott_to_you_fixed_mu_eta(willpos), {"alpha": radians(2)}
        )


###############################################################################


def willmott_to_you_fixed_mu_chi(pos):
    return Position(
        mu=-0,
        delta=pos.delta,
        nu=pos.gamma,
        eta=pos.omegah,
        chi=pi / 2,
        phi=-pos.phi,
    )


class TestUBCalculationWithWillmotStrategy_Si_5_5_12_FixedMuChi:
    def testAgainstResultsFromJan_27_2010(self):
        self.ubcalc = UBCalculation()
        self.ubcalc.set_lattice(
            "Si_5_5_12", [7.68, 53.48, 75.63, pi / 2, pi / 2, pi / 2]
        )
        self.ubcalc.add_reflection(
            HKL0,
            willmott_to_you_fixed_mu_chi(REF0),
            ENERGY,
            "ref0",
        )
        self.ubcalc.add_reflection(
            HKL1,
            willmott_to_you_fixed_mu_chi(REF1),
            ENERGY,
            "ref1",
        )
        self.ubcalc.calc_ub()
        print("U: ", self.ubcalc.U)
        print("UB: ", self.ubcalc.UB)
        matrixeq_(self.ubcalc.U, U_DIFFCALC)


class Test_Fixed_Mu_Chi(TestFixedMuEta):
    def _configure_constraints(self):
        self.hklcalc.constraints = Constraints(
            {"alpha": radians(2), "mu": 0, "chi": pi / 2}
        )

    def _convert_willmott_pos(self, willmott_pos):
        return willmott_to_you_fixed_mu_chi(willmott_pos)


def willmott_to_you_fixed_eta_chi(pos):
    return Position(
        mu=pos.omegah,
        delta=-pos.gamma,
        nu=pos.delta,
        eta=0,
        chi=0,
        phi=-pos.phi,
    )


class Test_Fixed_Eta_Chi(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)
        self.hklcalc.constraints = Constraints(
            {"alpha": radians(2), "chi": 0, "eta": 0}
        )
        self.wavelength = 0.6358

    def _convert_willmott_pos(self, willmott_pos):
        return willmott_to_you_fixed_eta_chi(willmott_pos)

    def testHkl_2_19_32_found_orientation_setting(self):
        pytest.skip()

    def testHkl_0_7_22_found_orientation_setting(self):
        pytest.skip()

    def testHkl_2_19_32_calculated_from_DDIF(self):
        pytest.skip()

    def testHkl_0_7_22_calculated_from_DDIF(self):
        pytest.skip()

    def testHkl_2_m5_12_calculated_from_DDIF(self):
        pytest.skip()

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice(
            "xtal", [7.68, 53.48, 75.63, pi / 2, pi / 2, pi / 2]
        )
        self.hklcalc.ubcalc.set_u(U_DIFFCALC)

    def _check(self, hkl, pos, virtual_expected={}):
        self._check_angles_to_hkl(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )
        self._check_hkl_to_angles(
            "", 999, 999, hkl, pos, self.wavelength, virtual_expected
        )

    def testHkl_2_19_32_calculated_predicted_with_diffcalc_and_found(self):
        willpos = WillPos(
            delta=radians(22.0332862),
            gamma=radians(-4.0973643),
            omegah=radians(2),
            phi=radians(64.0273584),
        )
        self._check(
            (2, 19, 32), self._convert_willmott_pos(willpos), {"alpha": radians(2)}
        )

    def testHkl_0_7_22_calculated_predicted_with_diffcalc_and_found(self):
        willpos = WillPos(
            delta=radians(11.2572236),
            gamma=radians(-2.9800571),
            omegah=radians(2),
            phi=radians(-86.5634425),
        )
        self._check(
            (0, 7, 22), self._convert_willmott_pos(willpos), {"alpha": radians(2)}
        )

    def testHkl_2_m5_12_calculated_predicted_with_diffcalc_and_found(self):
        willpos = WillPos(
            delta=radians(5.3109941),
            gamma=radians(-10.3716944),
            omegah=radians(2),
            phi=radians(167.0041454),
        )
        self._check(
            (2, -5, 12), self._convert_willmott_pos(willpos), {"alpha": radians(2)}
        )


# # Primary and secondary reflections found with the help of DDIF on Diamond's
# # i07 on Jan 28/29 2010


Pt531_HKL0 = -1.000, 1.000, 6.0000
Pt531_REF0 = WillPos(
    delta=radians(9.3971025),
    gamma=radians(16.1812303),
    omegah=radians(2),
    phi=radians(-52.1392905),
)

Pt531_HKL1 = -2.000, -1.000, 7.0000
Pt531_REF1 = WillPos(
    delta=radians(11.0126958),
    gamma=radians(-11.8636128),
    omegah=radians(2),
    phi=radians(40.3803393),
)
Pt531_REF12 = WillPos(
    delta=radians(11.0126958),
    gamma=radians(11.8636128),
    omegah=radians(2),
    phi=radians(-121.2155975),
)
Pt531_HKL2 = 1, 1, 9
Pt531_REF2 = WillPos(
    delta=radians(14.1881617),
    gamma=radians(7.7585939),
    omegah=radians(2),
    phi=radians(23.0203132),
)
Pt531_REF22 = WillPos(
    delta=radians(14.1881617),
    gamma=radians(-7.7585939),
    omegah=radians(2),
    phi=radians(-183.465146),
)
Pt531_WAVELENGTH = 0.6358

# # This is U matrix displayed by DDIF
U_FROM_DDIF = array(
    [
        [-0.00312594, -0.00063417, 0.99999491],
        [0.99999229, -0.00237817, 0.00312443],
        [0.00237618, 0.99999697, 0.00064159],
    ]
)

# # This is the version that Diffcalc comes up with ( see following test)
Pt531_U_DIFFCALC = array(
    [
        [-0.0023763, -0.9999970, -0.0006416],
        [0.9999923, -0.0023783, 0.0031244],
        [-0.0031259, -0.0006342, 0.9999949],
    ]
)


class TestUBCalculation_Pt531_FixedMuChi:
    def testAgainstResultsFromJan_28_2010(self):
        self.ubcalc = UBCalculation()
        self.ubcalc.set_lattice(
            "Pt531", [6.204, 4.806, 23.215, pi / 2, pi / 2, radians(49.8)]
        )

        self.ubcalc.add_reflection(
            Pt531_HKL0,
            willmott_to_you_fixed_mu_chi(Pt531_REF0),
            12.39842 / Pt531_WAVELENGTH,
            "ref0",
        )
        self.ubcalc.add_reflection(
            Pt531_HKL1,
            willmott_to_you_fixed_mu_chi(Pt531_REF1),
            12.39842 / Pt531_WAVELENGTH,
            "ref1",
        )
        self.ubcalc.calc_ub()
        print("U: ", self.ubcalc.U)
        print("UB: ", self.ubcalc.UB)
        matrixeq_(self.ubcalc.U, Pt531_U_DIFFCALC)


class Test_Pt531_FixedMuChi(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)
        self.hklcalc.constraints = Constraints(
            {"alpha": radians(2), "mu": 0, "chi": pi / 2}
        )
        self.wavelength = Pt531_WAVELENGTH
        CUT = Crystal(
            "Pt531", "Triclinic", 6.204, 4.806, 23.215, pi / 2, pi / 2, radians(49.8)
        )
        B = CUT.B
        self.UB = Pt531_U_DIFFCALC @ B

    def _convert_willmott_pos(self, willmott_pos):
        return willmott_to_you_fixed_mu_chi(willmott_pos)

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice(
            "Pt531", [6.204, 4.806, 23.215, pi / 2, pi / 2, radians(49.8)]
        )
        self.hklcalc.ubcalc.set_u(Pt531_U_DIFFCALC)

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

    def testHkl_0_found_orientation_setting(self):
        """Check that the or0 reflection maps back to the assumed hkl"""
        self.places = 1
        self._check_angles_to_hkl(
            "",
            999,
            999,
            Pt531_HKL0,
            self._convert_willmott_pos(Pt531_REF0),
            self.wavelength,
            {"alpha": radians(2)},
        )

    def testHkl_1_found_orientation_setting(self):
        """Check that the or1 reflection maps back to the assumed hkl"""
        self.places = 0
        self._check_angles_to_hkl(
            "",
            999,
            999,
            Pt531_HKL1,
            self._convert_willmott_pos(Pt531_REF1),
            self.wavelength,
            {"alpha": radians(2)},
        )

    def testHkl_0_calculated_from_DDIF(self):
        self.places = 7
        pos_expected = self._convert_willmott_pos(Pt531_REF0)
        self._check(Pt531_HKL0, pos_expected, {"alpha": radians(2)})

    def testHkl_1_calculated_from_DDIF(self):
        self.places = 7
        self._check(
            Pt531_HKL1, self._convert_willmott_pos(Pt531_REF1), {"alpha": radians(2)}
        )

    def testHkl_2_calculated_from_DDIF(self):
        self.places = 5
        self._check(
            Pt531_HKL2, self._convert_willmott_pos(Pt531_REF2), {"alpha": radians(2)}
        )

    def testHkl_2_m1_0_16(self):
        self.places = 5
        pos = WillPos(
            delta=radians(25.7990976),
            gamma=radians(-6.2413545),
            omegah=radians(2),
            phi=radians(47.4624380),
        )
        #        pos.phi -= 360
        self._check((-1, 0, 16), self._convert_willmott_pos(pos), {"alpha": radians(2)})


class Test_Pt531_Fixed_Mu_eta_(_BaseTest):
    def setup_method(self):
        _BaseTest.setup_method(self)
        self.hklcalc.constraints = Constraints(
            {"alpha": radians(2), "mu": -pi / 2, "eta": 0}
        )
        self.wavelength = Pt531_WAVELENGTH
        CUT = Crystal(
            "Pt531", "Triclinic", 6.204, 4.806, 23.215, pi / 2, pi / 2, radians(49.8)
        )
        B = CUT.B
        self.UB = Pt531_U_DIFFCALC @ B

    def _configure_ub(self):
        self.hklcalc.ubcalc.set_lattice(
            "Pt531", [6.204, 4.806, 23.215, pi / 2, pi / 2, radians(49.8)]
        )
        self.hklcalc.ubcalc.set_u(Pt531_U_DIFFCALC)

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

    def _convert_willmott_pos(self, willmott_pos):
        return willmott_to_you_fixed_mu_eta(willmott_pos)

    def testHkl_1_calculated_from_DDIF(self):
        self.places = 7
        self._check(
            Pt531_HKL1, self._convert_willmott_pos(Pt531_REF12), {"alpha": radians(2)}
        )

    def testHkl_2_calculated_from_DDIF(self):
        self.places = 5
        self._check(
            Pt531_HKL2, self._convert_willmott_pos(Pt531_REF22), {"alpha": radians(2)}
        )

    def testHkl_2_m1_0_16(self):
        self.places = 5
        pos = WillPos(
            delta=radians(25.7990976),
            gamma=radians(6.2413545),
            omegah=radians(2),
            phi=radians(-47.4949600),
        )
        #        pos.phi -= 360
        self._check((-1, 0, 16), self._convert_willmott_pos(pos), {"alpha": radians(2)})
