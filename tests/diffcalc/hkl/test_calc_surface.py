from math import pi
from typing import Dict, Union

import numpy as np
import pytest
from diffcalc import Q, ureg
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import Angle, DiffcalcException, I
from typing_extensions import Literal

from tests.diffcalc.hkl.test_calc import (
    Case,
    convert_position_to_hkl_and_hkl_to_position,
)
from tests.tools import matrixeq_


@pytest.fixture
def hklcalc() -> HklCalculation:
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore

    return HklCalculation(ubcalc, Constraints())


@pytest.fixture
def cubic() -> HklCalculation:
    ubcalc = UBCalculation()
    ubcalc.n_phi = (1, 0, 0)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore

    ubcalc.set_lattice("xtal", 1)
    ubcalc.set_u(I)

    return HklCalculation(ubcalc, Constraints())


@pytest.mark.parametrize(
    ("case", "expected_virtual"),
    [
        (
            Case("", (0, 0, 1), (-90, 60, 0, 0, 120, 90)),
            {"betain": 30, "betaout": 30},
        ),
        (
            Case("", (0, 1, 1), (-90, 90, 0, 0, 90, 90)),
            {"betain": 0, "betaout": 90},
        ),
        (
            Case("", (0, 1, 0), (-90, 60, 0, 0, 30, 90)),
            {"betain": -60, "betaout": 60},
        ),
        (
            Case("", (1, 1, 0), (-90, 90, 0, 0, 45, 45)),
            {"alpha": 30, "beta": 30},
        ),
    ],
)
def test_surface_normal_vertical_cubic(
    cubic: HklCalculation,
    case: Case,
    expected_virtual: Dict[str, Union[Angle, Literal["True"]]],
):
    cubic.constraints.asdict = {"a_eq_b": True, "mu": -pi / 2, "eta": 0}

    convert_position_to_hkl_and_hkl_to_position(cubic, case, 5, expected_virtual)


def test_surface_normal_vertical_cubic_fails_for_parallel_vectors(
    cubic: HklCalculation,
):
    cubic.constraints.asdict = {"a_eq_b": True, "mu": -pi / 2, "eta": 0}

    with pytest.raises(DiffcalcException):
        cubic.get_position(1, 0, 0, 1)


def test_ub_with_willmot_si_5_5_12_mu_eta_fixed():
    ubcalc = UBCalculation("test")
    ubcalc.set_lattice("Si_5_5_12", 7.68, 53.48, 75.63, 90, 90, 90)
    ubcalc.add_reflection(
        (2, 19, 32),
        Position(*(-90, 21.975, 4.419, 0.0, 92.0, -416.2) * ureg.deg),
        19.5005033,
        "ref0",
    )
    ubcalc.add_reflection(
        (0, 7, 22),
        Position(*(-90, 11.292, 2.844, 0, 92, -214.1) * ureg.deg),
        19.5005033,
        "ref1",
    )
    ubcalc.calc_ub()
    matrixeq_(
        ubcalc.U,
        np.array(
            [
                [-0.7178876, 0.6643924, -0.2078944],
                [-0.6559596, -0.5455572, 0.5216170],
                [0.2331402, 0.5108327, 0.8274634],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("case", "places"),
    [
        (
            Case("2 19 32", (2, 19, 32), (-90, 21.975, 4.419, 0, 92, -416.2), 0.6358),
            2,
        ),
        (
            Case("0 7 22", (0, 7, 22), (-90, 11.292, 2.844, 0, 92, -214.1), 0.6358),
            0,
        ),
        (
            Case("", (2, 19, 32), (-90.0, 21.974, 4.419, 0.0, 92.0, -56.197), 0.6358),
            3,
        ),
        (
            Case(
                "",
                (0, 7, 22),
                (-90.0, 11.241801854649, -3.038407637123, 0.0, 92.0, -3.43655749733),
                0.6358,
            ),
            3,
        ),
        (
            Case("", (2, -5, 12), (-90.0, 5.224, 10.415, 0.0, 92.0, -88.028), 0.6358),
            3,
        ),
        (
            Case(
                "",
                (2, 19, 32),
                (-90.0, 21.974032376045002, 4.418955754003, 0.0, 92.0, -56.19746),
                0.6358,
            ),
            5,
        ),
        (
            Case(
                "",
                (0, 7, 22),
                (-90.0, 11.241801854649, -3.038407637123, 0.0, 92.0, -3.43655749733),
                0.6358,
            ),
            5,
        ),
        (
            Case(
                "",
                (2, -5, 12),
                (-90.0, 5.223972025344, 10.415435905622, 0.0, 92.0, -88.02751),
                0.6358,
            ),
            5,
        ),
    ],
)
def test_fixed_mu_eta(hklcalc: HklCalculation, case: Case, places: int):
    hklcalc.constraints.asdict = {"alpha": Q(2, "deg"), "mu": -pi / 2, "eta": 0}
    hklcalc.ubcalc.set_lattice("xtal", 7.68, 53.48, 75.63, 90, 90, 90)
    hklcalc.ubcalc.set_u(
        np.array(
            [
                [-0.7178876, 0.6643924, -0.2078944],
                [-0.6559596, -0.5455572, 0.5216170],
                [0.2331402, 0.5108327, 0.8274634],
            ]
        )
    )

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, places, {"alpha": 2})


def test_ub_with_willmot_si_5_5_12_mu_chi_fixed():
    ubcalc = UBCalculation("test")
    ubcalc.set_lattice("Si_5_5_12", 7.68, 53.48, 75.63, 90, 90, 90)
    ubcalc.add_reflection(
        (2, 19, 32),
        Position(*(0.0, 21.975, 4.419, 2.0, 90.0, -326.2) * ureg.deg),
        19.5005033,
        "ref0",
    )
    ubcalc.add_reflection(
        (0, 7, 22),
        Position(*(0.0, 11.292, 2.844, 2.0, 90.0, -124.1) * ureg.deg),
        19.5005033,
        "ref1",
    )
    ubcalc.calc_ub()
    matrixeq_(
        ubcalc.U,
        np.array(
            [
                [-0.7178876, 0.6643924, -0.2078944],
                [-0.6559596, -0.5455572, 0.5216170],
                [0.2331402, 0.5108327, 0.8274634],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("case"),
    [
        Case(
            "", (2, 19, 32), (2.0, 4.0973643, 22.0332862, 0.0, 0.0, -64.0273584), 0.6358
        ),
        Case(
            "", (0, 7, 22), (2.0, 2.9800571, 11.2572236, 0.0, 0.0, 86.5634425), 0.6358
        ),
        Case(
            "",
            (2, -5, 12),
            (2.0, 10.3716944, 5.3109941, 0.0, 0.0, -167.00414540000003),
            0.6358,
        ),
    ],
)
def test_fixed_eta_chi(hklcalc: HklCalculation, case: Case):
    hklcalc.constraints.asdict = {"alpha": Q(2, "deg"), "eta": 0, "chi": 0}
    hklcalc.ubcalc.set_lattice("xtal", 7.68, 53.48, 75.63, 90, 90, 90)
    hklcalc.ubcalc.set_u(
        np.array(
            [
                [-0.7178876, 0.6643924, -0.2078944],
                [-0.6559596, -0.5455572, 0.5216170],
                [0.2331402, 0.5108327, 0.8274634],
            ]
        )
    )

    convert_position_to_hkl_and_hkl_to_position(
        hklcalc, case, 5, {"alpha": Q(2, "deg")}
    )


def test_ub_with_willmot_pt531_mu_chi_fixed():
    ubcalc = UBCalculation("test")
    ubcalc.set_lattice("Pt531", 6.204, 4.806, 23.215, 90, 90, 49.8)
    ubcalc.add_reflection(
        (-1, 1, 6),
        Position(
            *(0.0, 9.397102500000003, 16.1812303, 2.0, 90.0, 52.1392905) * ureg.deg
        ),
        19.5005033,
        "ref0",
    )
    ubcalc.add_reflection(
        (-2, -1, 7),
        Position(
            *(0.0, 11.012695800000001, -11.8636128, 2.0, 90.0, -40.3803393) * ureg.deg
        ),
        19.5005033,
        "ref1",
    )
    ubcalc.calc_ub()
    matrixeq_(
        ubcalc.U,
        np.array(
            [
                [-0.0023763, -0.9999970, -0.0006416],
                [0.9999923, -0.0023783, 0.0031244],
                [-0.0031259, -0.0006342, 0.9999949],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("case", "places"),
    [
        (
            Case(
                "",
                (-1, 1, 6),
                (0.0, 9.397102500000003, 16.1812303, 2.0, 90.0, 52.1392905),
                0.6358,
            ),
            1,
        ),
        (
            Case(
                "",
                (-2, -1, 7),
                (0.0, 11.012695800000001, -11.8636128, 2.0, 90.0, -40.3803393),
                0.6358,
            ),
            0,
        ),
        (
            Case(
                "",
                (-1, 1, 6),
                (0.0, 9.397102500000003, 16.1812303, 2.0, 90.0, 52.1392905),
                0.6358,
            ),
            7,
        ),
        (
            Case(
                "",
                (-2, -1, 7),
                (0.0, 11.012695800000001, -11.8636128, 2.0, 90.0, -40.3803393),
                0.6358,
            ),
            7,
        ),
        (
            Case(
                "",
                (1, 1, 9),
                (0.0, 14.1881617, 7.7585939, 2.0, 90.0, -23.0203132),
                0.6358,
            ),
            5,
        ),
        (
            Case(
                "",
                (-1, 0, 16),
                (0.0, 25.7990976, -6.2413545, 2.0, 90.0, -47.462438),
                0.6358,
            ),
            5,
        ),
    ],
)
def test_pt531_fixed_mu_chi(hklcalc: HklCalculation, case: Case, places: int):
    hklcalc.constraints.asdict = {"alpha": Q(2, "deg"), "mu": 0, "chi": pi / 2}
    hklcalc.ubcalc.set_lattice("Pt531", 6.204, 4.806, 23.215, 90, 90, 49.8)
    hklcalc.ubcalc.set_u(
        np.array(
            [
                [-0.0023763, -0.9999970, -0.0006416],
                [0.9999923, -0.0023783, 0.0031244],
                [-0.0031259, -0.0006342, 0.9999949],
            ]
        )
    )

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, places, {"alpha": 2})


@pytest.mark.parametrize(
    ("case", "places"),
    [
        (
            Case(
                "",
                (-2, -1, 7),
                (-90.0, 11.012695800000001, 11.8636128, 0.0, 92.0, 31.2155975),
                0.6358,
            ),
            7,
        ),
        (
            Case(
                "",
                (1, 1, 9),
                (-90.0, 14.1881617, -7.7585939, 0.0, 92.0, 93.465146),
                0.6358,
            ),
            5,
        ),
        (
            Case(
                "",
                (-1, 0, 16),
                (-90.0, 25.7990976, 6.2413545, 0.0, 92.0, -42.50504),
                0.6358,
            ),
            5,
        ),
    ],
)
def test_pt531_fixed_mu_eta(hklcalc: HklCalculation, case: Case, places: int):
    hklcalc.constraints.asdict = {"alpha": Q(2, "deg"), "mu": -pi / 2, "eta": 0}

    hklcalc.ubcalc.set_lattice("Pt531", 6.204, 4.806, 23.215, 90, 90, 49.8)
    hklcalc.ubcalc.set_u(
        np.array(
            [
                [-0.0023763, -0.9999970, -0.0006416],
                [0.9999923, -0.0023783, 0.0031244],
                [-0.0031259, -0.0006342, 0.9999949],
            ]
        )
    )

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, places, {"alpha": 2})
