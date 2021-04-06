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

import math
from unittest.mock import Mock

from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import TODEG, I
from numpy import array

from tests.diffcalc.scenarios import Pos
from tests.tools import assert_almost_equal, assert_matrix_almost_equal

x = array([[1], [0], [0]])
y = array([[0], [1], [0]])
z = array([[0], [0], [1]])


def isnan(n):
    # math.isnan was introduced only in python 2.6 and is not in Jython (2.5.2)
    try:
        return math.isnan(n)
    except AttributeError:
        return n != n  # for Jython


class Test_position_to_virtual_angles:
    def setup_method(self):
        constraints = Mock()
        constraints.is_fully_constrained.return_value = True
        self.ubcalc = UBCalculation()
        self.ubcalc.set_lattice("xtal", 1)
        self.ubcalc.set_u(I)
        self.ubcalc.n_phi = (0, 0, 1)
        self.calc = HklCalculation(self.ubcalc, constraints)

    def check_angle(
        self, name, expected, mu=-99, delta=99, nu=99, eta=99, chi=99, phi=99
    ):
        """All in degrees"""
        pos = Pos(mu=mu, delta=delta, nu=nu, eta=eta, chi=chi, phi=phi)
        calculated = self.calc.get_virtual_angles(pos)[name] * TODEG
        assert_almost_equal(calculated, expected)

    # theta

    def test_theta0(self):
        self.check_angle("theta", 0, delta=0, nu=0)

    def test_theta1(self):
        self.check_angle("theta", 1, delta=2, nu=0)

    def test_theta2(self):
        self.check_angle("theta", 1, delta=0, nu=2)

    def test_theta3(self):
        self.check_angle("theta", 1, delta=-2, nu=0)

    def test_theta4(self):
        self.check_angle("theta", 1, delta=0, nu=-2)

    # qaz

    def test_qaz0_degenerate_case(self):
        self.check_angle("qaz", 0, delta=0, nu=0)

    def test_qaz1(self):
        self.check_angle("qaz", 90, delta=2, nu=0)

    def test_qaz2(self):
        self.check_angle("qaz", 90, delta=90, nu=0)

    def test_qaz3(self):
        self.check_angle(
            "qaz",
            0,
            delta=0,
            nu=1,
        )

    # Can't see one by eye
    # def test_qaz4(self):
    #    pos = Pos(delta=20*TORAD, nu=20*TORAD)#.inRadians()
    #    assert_almost_equal(
    #        self.calc._anglesToVirtualAngles(pos, None)['qaz']*TODEG, 45)

    # alpha
    def test_defaultReferenceValue(self):
        # The following tests depemd on this
        assert_matrix_almost_equal(self.calc.ubcalc.n_phi, array([[0], [0], [1]]))

    def test_alpha0(self):
        self.check_angle("alpha", 0, mu=0, eta=0, chi=0, phi=0)

    def test_alpha1(self):
        self.check_angle("alpha", 0, mu=0, eta=0, chi=0, phi=10)

    def test_alpha2(self):
        self.check_angle("alpha", 0, mu=0, eta=0, chi=0, phi=-10)

    def test_alpha3(self):
        self.check_angle("alpha", 2, mu=2, eta=0, chi=0, phi=0)

    def test_alpha4(self):
        self.check_angle("alpha", -2, mu=-2, eta=0, chi=0, phi=0)

    def test_alpha5(self):
        self.check_angle("alpha", 2, mu=0, eta=90, chi=2, phi=0)

    # beta

    def test_beta0(self):
        self.check_angle("beta", 0, delta=0, nu=0, mu=0, eta=0, chi=0, phi=0)

    def test_beta1(self):
        self.check_angle("beta", 0, delta=10, nu=0, mu=0, eta=6, chi=0, phi=5)

    def test_beta2(self):
        self.check_angle("beta", 10, delta=0, nu=10, mu=0, eta=0, chi=0, phi=0)

    def test_beta3(self):
        self.check_angle("beta", -10, delta=0, nu=-10, mu=0, eta=0, chi=0, phi=0)

    def test_beta4(self):
        self.check_angle("beta", 5, delta=0, nu=10, mu=5, eta=0, chi=0, phi=0)

    # azimuth
    def test_naz0(self):
        self.check_angle("naz", 0, mu=0, eta=0, chi=0, phi=0)

    def test_naz1(self):
        self.check_angle("naz", 0, mu=0, eta=0, chi=0, phi=10)

    def test_naz3(self):
        self.check_angle("naz", 0, mu=10, eta=0, chi=0, phi=10)

    def test_naz4(self):
        self.check_angle("naz", 2, mu=0, eta=0, chi=2, phi=0)

    def test_naz5(self):
        self.check_angle("naz", -2, mu=0, eta=0, chi=-2, phi=0)

    # tau
    def test_tau0(self):
        self.check_angle("tau", 0, mu=0, delta=0, nu=0, eta=0, chi=0, phi=0)
        # self.check_angle('tau_from_dot_product', 90, mu=0, delta=0,
        # nu=0, eta=0, chi=0, phi=0)

    def test_tau1(self):
        self.check_angle("tau", 90, mu=0, delta=20, nu=0, eta=10, chi=0, phi=0)
        # self.check_angle('tau_from_dot_product', 90, mu=0, delta=20,
        # nu=0, eta=10, chi=0, phi=0)

    def test_tau2(self):
        self.check_angle("tau", 90, mu=0, delta=20, nu=0, eta=10, chi=0, phi=3)
        # self.check_angle('tau_from_dot_product', 90, mu=0, delta=20,
        # nu=0, eta=10, chi=0, phi=3)

    def test_tau3(self):
        self.check_angle("tau", 88, mu=0, delta=20, nu=0, eta=10, chi=2, phi=0)
        # self.check_angle('tau_from_dot_product', 88, mu=0, delta=20,
        # nu=0, eta=10, chi=2, phi=0)

    def test_tau4(self):
        self.check_angle("tau", 92, mu=0, delta=20, nu=0, eta=10, chi=-2, phi=0)
        # self.check_angle('tau_from_dot_product', 92, mu=0, delta=20,
        # nu=0, eta=10, chi=-2, phi=0)

    def test_tau5(self):
        self.check_angle("tau", 10, mu=0, delta=0, nu=20, eta=0, chi=0, phi=0)
        # self.check_angle('tau_from_dot_product', 10, mu=0, delta=0,
        # nu=20, eta=0, chi=0, phi=0)

    # psi

    def test_psi0(self):
        pos = Position()
        assert isnan(self.calc.get_virtual_angles(pos)["psi"])

    def test_psi1(self):
        self.check_angle("psi", 90, mu=0, delta=11, nu=0, eta=0, chi=0, phi=0)

    def test_psi2(self):
        self.check_angle("psi", 100, mu=10, delta=0.001, nu=0, eta=0, chi=0, phi=0)

    def test_psi3(self):
        self.check_angle("psi", 80, mu=-10, delta=0.001, nu=0, eta=0, chi=0, phi=0)

    def test_psi4(self):
        self.check_angle("psi", 90, mu=0, delta=11, nu=0, eta=0, chi=0, phi=12.3)

    def test_psi5(self):
        # self.check_angle('psi', 0, mu=10, delta=.00000001,
        # nu=0, eta=0, chi=90, phi=0)
        pos = Pos(mu=0, delta=0, nu=0, eta=0, chi=90, phi=0)
        assert isnan(self.calc.get_virtual_angles(pos)["psi"])

    def test_psi6(self):
        self.check_angle("psi", 90, mu=0, delta=0.001, nu=0, eta=90, chi=0, phi=0)

    def test_psi7(self):
        self.check_angle("psi", 92, mu=0, delta=0.001, nu=0, eta=90, chi=2, phi=0)

    def test_psi8(self):
        self.check_angle("psi", 88, mu=0, delta=0.001, nu=0, eta=90, chi=-2, phi=0)
