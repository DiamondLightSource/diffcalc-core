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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the*
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Diffcalc.  If not, see <http://www.gnu.org/licenses/>.
###

from math import asin, atan2, cos, pi, radians, sin

# from diffcalc.hkl.vlieg.geometry import VliegPosition
from diffcalc.hkl.calc import sign
from diffcalc.hkl.geometry import Position
from diffcalc.ub.reference import Reflection


def PosFromI16sEuler(phi, chi, eta, mu, delta, gamma):
    return Position(
        mu=mu,
        delta=delta,
        nu=gamma,
        eta=eta,
        chi=chi,
        phi=phi,
    )


def VliegPos(alpha=None, delta=None, gamma=None, omega=None, chi=None, phi=None):
    """Convert six-circle Vlieg diffractometer angles into 4S+2D You geometry"""
    sin_alpha = sin(alpha)
    cos_alpha = cos(alpha)
    sin_delta = sin(delta)
    cos_delta = cos(delta)
    sin_gamma = sin(gamma)
    cos_gamma = cos(gamma)  # this was, before, assuming args were in degrees.
    asin_delta = asin(sin_delta * cos_gamma)  # Eq.(83)
    vals_delta = [asin_delta, pi - asin_delta]
    idx, _ = min(
        ((i, abs(delta - d)) for i, d in enumerate(vals_delta)), key=lambda x: x[1]
    )
    pos_delta = vals_delta[idx]
    sgn = sign(cos(pos_delta))
    pos_nu = atan2(
        sgn * (cos_delta * cos_gamma * sin_alpha + cos_alpha * sin_gamma),
        sgn * (cos_delta * cos_gamma * cos_alpha - sin_alpha * sin_gamma),
    )  # Eq.(84)

    return Position(mu=alpha, delta=pos_delta, nu=pos_nu, eta=omega, chi=chi, phi=phi)


class SessionScenario:
    """
    A test scenario. The test case must have __name, lattice and bmatrix set
    and if umatrix is set then so must ref 1 and ref 2. Matrices should be 3*3
    python arrays of lists and ref1 and ref2 in the format (h, k, l, position,
    energy, tag)."""

    def __init__(self):
        self.name = None
        self.lattice = None
        self.bmatrix = None
        self.ref1 = None
        self.ref2 = None
        self.umatrix = None
        self.calculations = []  # CalculationScenarios

    def __str__(self):
        toReturn = "\nTestScenario:"
        toReturn += "\n     name: " + self.name
        toReturn += "\n  lattice:" + str(self.lattice)
        toReturn += "\n  bmatrix:" + str(self.bmatrix)
        toReturn += "\n     ref1:" + str(self.ref1)
        toReturn += "\n     ref2:" + str(self.ref2)
        toReturn += "\n  umatrix:" + str(self.umatrix)
        return toReturn


class CalculationScenario:
    """
    Used as part of a test scenario. A UB matrix appropriate for this
    calcaultion will have been calculated or loaded
    """

    def __init__(self, tag, package, mode, energy, modeToTest, modeNumber):
        self.tag = tag
        self.package = package
        self.mode = mode
        self.energy = energy
        self.wavelength = 12.39842 / energy
        self.modeToTest = modeToTest
        self.modeNumber = modeNumber
        self.hklList = None  # hkl triples
        self.posList = []
        self.paramList = []


def sessions(P=VliegPos):
    ############################ SESSION0 ############################
    # From the dif_init.mat next to dif_dos.exe on Vlieg'session2 cd
    # session2 = SessionScenario()
    # session2.name = 'latt1'
    # session2.lattice = ([4.0004, 4.0004, 2.270000, 90, 90, 90])
    # session2.bmatrix = (((1.570639, 0, 0) ,(0.0, 1.570639, 0) ,
    #                      (0.0, 0.0, 2.767923)))
    # self.scenarios.append(session2)

    ############################ SESSION1 ############################
    # From b16 on 27June2008 (From Chris Nicklin)

    session1 = SessionScenario()
    session1.name = "b16_270608"
    session1.lattice = (3.8401, 3.8401, 5.43072, pi / 2, pi / 2, pi / 2)
    session1.bmatrix = ((1.636204, 0, 0), (0, 1.636204, 0), (0, 0, 1.156971))
    session1.ref1 = Reflection(
        1,
        0,
        1.0628,
        P(
            radians(5.000),
            radians(22.790),
            0.000,
            radians(1.552),
            radians(22.400),
            radians(14.255),
        ),
        10,
        "ref1",
    )
    session1.ref2 = Reflection(
        0,
        1,
        1.0628,
        P(
            radians(5.000),
            radians(22.790),
            0.000,
            radians(4.575),
            radians(24.275),
            radians(101.320),
        ),
        10,
        "ref2",
    )
    session1.umatrix = (
        (0.997161, -0.062217, 0.042420),
        (0.062542, 0.998022, -0.006371),
        (-0.041940, 0.009006, 0.999080),
    )
    session1.ref1calchkl = (1, 0, 1.0628)  # Must match the guessed value!
    session1.ref2calchkl = (-0.0329, 1.0114, 1.04)

    ############################ SESSION2 ############################
    # cubic crystal from bliss tutorial
    session2 = SessionScenario()
    session2.name = "cubic_from_bliss_tutorial"
    session2.lattice = (1.54, 1.54, 1.54, pi / 2, pi / 2, pi / 2)
    session2.ref1 = Reflection(
        1, 0, 0, P(0, pi / 3, 0, pi / 6, 0, 0), 12.39842 / 1.54, "ref1"
    )
    session2.ref2 = Reflection(
        0, 1, 0, P(0, pi / 3, 0, pi / 6, 0, -pi / 2), 12.39842 / 1.54, "ref2"
    )
    session2.bmatrix = ((4.07999, 0, 0), (0, 4.07999, 0), (0, 0, 4.07999))
    session2.umatrix = ((1, 0, 0), (0, -1, 0), (0, 0, -1))
    session2.ref1calchkl = (1, 0, 0)  # Must match the guessed value!
    session2.ref2calchkl = (0, 1, 0)
    # sixc-0a : fixed omega = 0
    c = CalculationScenario("sixc-0a", "sixc", "0", 12.39842 / 1.54, "4cBeq", 1)
    c.alpha = 0
    c.gamma = 0
    c.w = 0
    # c.hklList=((0.7, 0.9, 1.3), (1,0,0), (0,1,0), (1, 1, 0))
    c.hklList = ((0.7, 0.9, 1.3),)
    c.posList.append(
        P(
            0.000000,
            radians(119.669750),
            0.000000,
            radians(59.834875),
            radians(-48.747500),
            radians(307.874983651098),
        )
    )
    # c.posList.append(P(0.000000, 60.000000, 0.000000, 30.000, 0.000000, 0.000000))
    # c.posList.append(P(0.000000, 60.000000, 0.000000, 30.000, 0.000000, -90.0000))
    # c.posList.append(P(0.000000, 90.000000, 0.000000, 45.000, 0.000000, -45.0000))
    session2.calculations.append(c)

    ############################ SESSION3 ############################
    # AngleCalc scenarios from SPEC sixc. using crystal and alignment
    session3 = SessionScenario()
    session3.name = "spec_sixc_b16_270608"
    session3.lattice = (3.8401, 3.8401, 5.43072, pi / 2, pi / 2, pi / 2)
    session3.bmatrix = ((1.636204, 0, 0), (0, 1.636204, 0), (0, 0, 1.156971))
    session3.umatrix = (
        (0.997161, -0.062217, 0.042420),
        (0.062542, 0.998022, -0.006371),
        (-0.041940, 0.009006, 0.999080),
    )
    session3.ref1 = Reflection(
        1,
        0,
        1.0628,
        P(
            radians(5.000),
            radians(22.790),
            0.000,
            radians(1.552),
            radians(22.400),
            radians(14.255),
        ),
        12.39842 / 1.24,
        "ref1",
    )
    session3.ref2 = Reflection(
        0,
        1,
        1.0628,
        P(
            radians(5.000),
            radians(22.790),
            0.000,
            radians(4.575),
            radians(24.275),
            radians(101.320),
        ),
        12.39842 / 1.24,
        "ref2",
    )
    session3.ref1calchkl = (1, 0, 1.0628)
    session3.ref2calchkl = (-0.0329, 1.0114, 1.04)
    # sixc-0a : fixed omega = 0
    ac = CalculationScenario("sixc-0a", "sixc", "0", 12.39842 / 1.24, "4cBeq", 1)
    ac.alpha = 0
    ac.gamma = 0
    ac.w = 0
    ### with 'omega_low':-90, 'omega_high':270, 'phi_low':-180, 'phi_high':180
    ac.hklList = []
    ac.hklList.append((0.7, 0.9, 1.3))
    ac.posList.append(
        P(
            0.0,
            radians(27.352179),
            0.000000,
            radians(13.676090),
            radians(37.774500),
            radians(53.965500),
        )
    )
    ac.paramList.append(
        {
            "Bin": radians(8.3284),
            "Bout": radians(8.3284),
            "rho": radians(36.5258),
            "eta": radians(0.1117),
            "twotheta": radians(27.3557),
        }
    )

    ac.hklList.append((1, 0, 0))
    ac.posList.append(
        P(
            0.0,
            radians(18.580230),
            0.000000,
            radians(9.290115),
            radians(-2.403500),
            radians(3.589000),
        )
    )
    ac.paramList.append(
        {
            "Bin": radians(-0.3880),
            "Bout": radians(-0.3880),
            "rho": radians(-2.3721),
            "eta": radians(-0.0089),
            "twotheta": radians(18.5826),
        }
    )

    ac.hklList.append((0, 1, 0))
    ac.posList.append(
        P(
            0.0,
            radians(18.580230),
            0.000000,
            radians(9.290115),
            radians(0.516000),
            radians(93.567000),
        )
    )
    ac.paramList.append(
        {
            "Bin": radians(0.0833),
            "Bout": radians(0.0833),
            "rho": radians(0.5092),
            "eta": radians(-0.0414),
            "twotheta": radians(18.5826),
        }
    )

    ac.hklList.append((1, 1, 0))
    ac.posList.append(
        P(
            0.0,
            radians(26.394192),
            0.000000,
            radians(13.197096),
            radians(-1.334500),
            radians(48.602000),
        )
    )
    ac.paramList.append(
        {
            "Bin": radians(-0.3047),
            "Bout": radians(-0.3047),
            "rho": radians(-1.2992),
            "eta": radians(-0.0351),
            "twotheta": radians(26.3976),
        }
    )

    session3.calculations.append(ac)

    ############################ SESSION4 ############################
    # test crystal

    session4 = SessionScenario()
    session4.name = "test_orth"
    session4.lattice = (1.41421, 1.41421, 1.00000, pi / 2, pi / 2, pi / 2)
    session4.system = "Orthorhombic"
    session4.bmatrix = ((4.44288, 0, 0), (0, 4.44288, 0), (0, 0, 6.28319))
    session4.ref1 = Reflection(
        0,
        1,
        2,
        P(0.0000, radians(122.4938), 0.0000, radians(80.7181), pi / 2, -pi / 4),
        15.0,
        "ref1",
    )
    session4.ref2 = Reflection(
        1,
        0,
        2,
        P(
            0.0000,
            radians(122.4938),
            0.000,
            radians(61.2469),
            radians(70.5288),
            -pi / 4,
        ),
        15,
        "ref2",
    )
    session4.ref3 = Reflection(
        1,
        0,
        1,
        P(0.0000, radians(60.8172), 0.000, radians(30.4086), radians(54.7356), -pi / 4),
        15,
        "ref3",
    )
    session4.ref4 = Reflection(
        1,
        1,
        2,
        P(0.0000, radians(135.0736), 0.000, radians(67.5368), radians(63.4349), 0.0000),
        15,
        "ref4",
    )
    session4.reflist = (session4.ref1, session4.ref2, session4.ref3, session4.ref4)
    session4.umatrix = (
        (0.70711, 0.70711, 0.00),
        (-0.70711, 0.70711, 0.00),
        (0.00, 0.00, 1.00),
    )
    session4.ref1calchkl = (0, 1, 2)  # Must match the guessed value!
    session4.ref2calchkl = (1, 0, 2)

    ############################ SESSION5 ############################
    # test crystal

    session5 = SessionScenario()
    session5.name = "Dalyite"
    session5.lattice = (7.51, 7.73, 7.00, radians(106.0), radians(113.5), radians(99.5))
    session5.system = "Triclinic"
    session5.bmatrix = (
        (0.96021, 0.27759, 0.49527),
        (0, 0.84559, 0.25738),
        (0, 0, 0.89760),
    )
    session5.ref1 = Reflection(
        0,
        1,
        2,
        P(
            0.0000,
            radians(23.7405),
            0.0000,
            radians(11.8703),
            radians(46.3100),
            radians(43.1304),
        ),
        12.3984,
        "ref1",
    )
    session5.ref2 = Reflection(
        1,
        0,
        3,
        P(
            0.0000,
            radians(34.4282),
            0.000,
            radians(17.2141),
            radians(46.4799),
            radians(12.7852),
        ),
        12.3984,
        "ref2",
    )
    session5.ref3 = Reflection(
        2,
        2,
        6,
        P(
            0.0000,
            radians(82.8618),
            0.000,
            radians(41.4309),
            radians(41.5154),
            radians(26.9317),
        ),
        12.3984,
        "ref3",
    )
    session5.ref4 = Reflection(
        4,
        1,
        4,
        P(
            0.0000,
            radians(71.2763),
            0.000,
            radians(35.6382),
            radians(29.5042),
            radians(14.5490),
        ),
        12.3984,
        "ref4",
    )
    session5.ref5 = Reflection(
        8,
        3,
        1,
        P(
            0.0000,
            radians(97.8850),
            0.000,
            radians(48.9425),
            radians(5.6693),
            radians(16.7929),
        ),
        12.3984,
        "ref5",
    )
    session5.ref6 = Reflection(
        6,
        4,
        5,
        P(
            0.0000,
            radians(129.6412),
            0.000,
            radians(64.8206),
            radians(24.1442),
            radians(24.6058),
        ),
        12.3984,
        "ref6",
    )
    session5.ref7 = Reflection(
        3,
        5,
        7,
        P(
            0.0000,
            radians(135.9159),
            0.000,
            radians(67.9579),
            radians(34.3696),
            radians(35.1816),
        ),
        12.3984,
        "ref7",
    )
    session5.reflist = (
        session5.ref1,
        session5.ref2,
        session5.ref3,
        session5.ref4,
        session5.ref5,
        session5.ref6,
        session5.ref7,
    )
    session5.umatrix = (
        (0.99982, 0.00073, 0.01903),
        (0.00073, 0.99710, -0.07612),
        (-0.01903, 0.07612, 0.99692),
    )
    session5.ref1calchkl = (0, 1, 2)  # Must match the guessed value!
    session5.ref2calchkl = (1, 0, 3)

    ############################ SESSION6 ############################
    # test crystal

    session6 = SessionScenario()
    session6.name = "Acanthite"
    session6.lattice = (4.229, 6.931, 7.862, pi / 2, radians(99.61), pi / 2)
    session6.system = "Monoclinic"
    session6.bmatrix = (
        (1.50688, 0.00000, 0.13532),
        (0.00000, 0.90653, 0.00000),
        (0.00000, 0.00000, 0.79918),
    )
    session6.ref1 = Reflection(
        0,
        1,
        2,
        P(
            0.0000,
            radians(21.1188),
            0.0000,
            radians(10.5594),
            radians(59.6447),
            radians(61.8432),
        ),
        10.0,
        "ref1",
    )
    session6.ref2 = Reflection(
        1,
        0,
        3,
        P(
            0.0000,
            radians(35.2291),
            0.000,
            radians(62.4207),
            radians(87.1516),
            radians(-90.0452),
        ),
        10.0,
        "ref2",
    )
    session6.ref3 = Reflection(
        1,
        1,
        6,
        P(
            0.0000,
            radians(64.4264),
            0.000,
            radians(63.9009),
            radians(97.7940),
            radians(-88.8808),
        ),
        10.0,
        "ref3",
    )
    session6.ref4 = Reflection(
        1,
        2,
        2,
        P(
            0.0000,
            radians(34.4369),
            0.000,
            radians(72.4159),
            radians(60.1129),
            radians(-29.0329),
        ),
        10.0,
        "ref4",
    )
    session6.ref5 = Reflection(
        2,
        2,
        1,
        P(
            0.0000,
            radians(43.0718),
            0.000,
            radians(21.5359),
            radians(8.3873),
            radians(29.0230),
        ),
        10.0,
        "ref5",
    )
    session6.reflist = (
        session6.ref1,
        session6.ref2,
        session6.ref3,
        session6.ref4,
        session6.ref5,
    )
    session6.umatrix = (
        (0.99411, 0.00079, 0.10835),
        (0.00460, 0.99876, -0.04949),
        (-0.10825, 0.04969, 0.99288),
    )
    session6.ref1calchkl = (0, 1, 2)  # Must match the guessed value!
    session6.ref2calchkl = (1, 0, 3)
    ########################################################################
    return (session1, session2, session3, session4, session5, session6)
