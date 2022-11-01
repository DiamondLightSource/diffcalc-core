import itertools
from math import cos, pi, radians, sin
from typing import Dict, Tuple, Union

import numpy as np
import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.ub.calc import UBCalculation
from diffcalc.ub.crystal import LatticeParams
from diffcalc.util import DiffcalcException, I

from tests.diffcalc import Q
from tests.diffcalc.hkl.test_calc import (
    Case,
    configure_ub,
    convert_position_to_hkl_and_hkl_to_position,
)


@pytest.fixture
def hklcalc() -> HklCalculation:
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore
    return HklCalculation(ubcalc, Constraints())


@pytest.fixture
def cubic() -> HklCalculation:
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore

    ubcalc.set_lattice("Cubic", LatticeParams(1))
    configure_ub(ubcalc)

    return HklCalculation(ubcalc, Constraints())


@pytest.mark.parametrize(
    ("expected_position", "constraints"),
    [
        ((0, -90, 0, -180, 90, 0), {"chi": pi / 2, "phi": 0, "psi": 0}),
        ((90, 90, 0, 0, 90, -90), {"mu": pi / 2, "eta": 0, "psi": 0}),
        ((0, 0, 90, 0, 0, -180), {"chi": 0, "eta": 0, "psi": 0}),
        ((0, 90, 0, 0, 90, -180), {"chi": pi / 2, "mu": 0, "psi": 0}),
        ((0, 0, 90, 90, 0, 90), {"mu": 0, "phi": pi / 2, "psi": 0}),
        ((90, -90, 0, 0, -90, 90), {"eta": 0, "phi": pi / 2, "psi": 0}),
        ((-45, -90, 0, -180, 90, 45), {"chi": pi / 2, "phi": pi / 4, "betain": 0}),
    ],
)
def test_get_position_one_ref_two_samp(
    cubic: HklCalculation,
    expected_position: Tuple[float, float, float, float, float, float],
    constraints: Dict[str, Union[float, bool]],
):
    cubic.constraints = Constraints(constraints)

    all_positions = cubic.get_position(0, 1, 1, 1)

    calculated_pos = all_positions[0][0]

    assert np.all(
        [
            radians(expected_position[i])
            == pytest.approx(float(calculated_pos.astuple[i]))
            for i in range(6)
        ]
    )


@pytest.mark.parametrize(
    ("case"),
    (
        Case("010", (0, 1, 0), (120, 0, 60, 0, 90, 0)),
        Case("011", (0, 1, 1), (45, 0, -90, 135, 90, 0)),
        Case("100", (1, 0, 0), (-30, 0, -60, 0, 90, 0)),
        Case("101", (1, 0, 1), (-30, 54.7356, -90, 35.2644, 90, 0)),
        Case("1 1 0.0001", (1, 1, 0.0001), (0, -0.00286, -90, 179.997, 90, 0)),
        Case("111", (1, 1, 1), (-171.5789, 20.9410, 122.3684, -30.3612, 90, 0)),
        Case("1.1 0 0", (1.1, 0, 0), (-146.6330, 0, 66.7340, 0, 90, 0)),
        Case("0.9 0 0", (0.9, 0, 0), (-153.2563, 0, 53.4874, 0, 90, 0)),
        Case(
            "0.7 0.8 0.8",
            (0.7, 0.8, 0.8),
            (167.7652, 23.7336, 82.7832, -24.1606, 90, 0),
        ),
        Case(
            "0.7 0.8 0.9",
            (0.7, 0.8, 0.9),
            (169.0428, 25.6713, 88.0926, -27.2811, 90, 0),
        ),
        Case("0.7 0.8 1", (0.7, 0.8, 1), (170.5280, 27.1595, 94.1895, -30.4583, 90, 0)),
    ),
)
def test_fixed_chi_phi_a_eq_b(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"chi": pi / 2, "phi": 0, "a_eq_b": True})

    convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


def test_fixed_chi_phi_a_eq_b_fails_for_non_unique_eta(cubic: HklCalculation):
    cubic.constraints = Constraints({"chi": pi / 2, "phi": 0, "a_eq_b": True})
    case = Case("1m10", (1, -1, 0), (-90, 0, 90, 0, 90, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


def test_fixed_chi_phi_a_eq_b_fails_for_parallel_vector(cubic: HklCalculation):
    cubic.constraints = Constraints({"chi": pi / 2, "phi": 0, "a_eq_b": True})
    case = Case("001", (0, 0, 1), (30, 0, 60, 90, 0, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


@pytest.mark.parametrize(
    ("case"),
    (
        Case(
            "100",
            (1, 0, 0),
            (0, 60, 0, 30, 0, 0),
        ),
        Case(
            "010-->100",
            (sin(radians(4)), cos(radians(4)), 0),
            (0, 60, 0, 120 - 4, 0, 0),
        ),
        Case("010", (0, 1, 0), (0, 60, 0, 120, 0, 0)),
        Case(
            "0.1 0 1.5",
            (0.1, 0, 1.5),
            (
                0,
                97.46959231642,
                0,
                97.46959231642 / 2,
                86.18592516571,
                0,
            ),
        ),
        Case(
            "001-->100",
            (cos(radians(86)), 0, sin(radians(86))),
            (0, 60, 0, 30, 90 - 4, 0),
        ),
    ),
)
def test_with_one_ref_and_mu_0_phi_0(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"psi": pi / 2, "mu": 0, "phi": 0})

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


def test_with_one_ref_and_mu_0_phi_0_fails_for_parallel_vectors(cubic: HklCalculation):
    cubic.constraints = Constraints({"psi": pi / 2, "mu": 0, "phi": 0})
    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case"),
    (
        Case(
            "100",
            (1, 0, 0),
            (-90, 60, 0, 0, 30, 0),
        ),
        Case(
            "100-->001",
            (cos(radians(4)), 0, sin(radians(4))),
            (-90, 60, 0, 0, 30 + 4, 0),
        ),
        Case("010", (0, 1, 0), (120, 0, 60, 0, 180, 0)),
        Case(
            "0.1 0 0.15",
            (0.1, 0, 0.15),
            (
                -90,
                10.34318,
                0,
                0,
                61.48152,
                0,
            ),
        ),
        Case(
            "010-->001",
            (0, cos(radians(4)), sin(radians(4))),
            (120 + 4, 0, 60, 0, 180, 0),
        ),
    ),
)
def test_with_one_ref_and_eta_0_phi_0(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"psi": 0, "eta": 0, "phi": 0})

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


def test_with_one_ref_and_eta_0_phi_0_fails_for_parallel_vectors(cubic: HklCalculation):
    cubic.constraints = Constraints({"psi": 0, "eta": 0, "phi": 0})
    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case", "places"),
    [
        (Case("010", (0, 1, 0), (0, 60, 0, 120, 0, 0)), 4),
        (Case("011", (0, 1, 1), (30, 54.7356, 90, 125.2644, 0, 0)), 4),
        (Case("100", (1, 0, 0), (0, 60, 0, 30, 0, 0)), 4),
        (Case("101", (1, 0, 1), (30, 54.7356, 90, 35.2644, 0, 0)), 4),
        (Case("110", (1, 1, 0), (0, 90, 0, 90, 0, 0)), 4),
        (Case("1 1 0.0001", (1, 1, 0.0001), (0.0029, 89.9971, 90.0058, 90, 0, 0)), 3),
        (Case("111", (1, 1, 1), (30, 54.7356, 150, 99.7356, 0, 0)), 4),
        (Case("1.1 0 0", (1.1, 0, 0), (0, 66.7340, 0, 33.3670, 0, 0)), 4),
        (Case("0.9 0 0", (0.9, 0, 0), (0, 53.4874, 0, 26.7437, 0, 0)), 4),
        (
            Case(
                "0.7 0.8 0.8",
                (0.7, 0.8, 0.8),
                (23.5782, 59.9980, 76.7037, 84.2591, 0, 0),
            ),
            4,
        ),
        (
            Case(
                "0.7 0.8 0.9",
                (0.7, 0.8, 0.9),
                (26.74368, 58.6754, 86.6919, 85.3391, 0, 0),
            ),
            4,
        ),
        (Case("0.7 0.8 1", (0.7, 0.8, 1), (30, 57.0626, 96.86590, 86.6739, 0, 0)), 4),
    ],
)
def test_i07_horizontal(hklcalc: HklCalculation, case: Case, places: int):
    hklcalc.ubcalc.set_lattice("Cubic", LatticeParams(1))
    hklcalc.constraints.asdict = {"chi": 0, "phi": 0, "a_eq_b": True}
    hklcalc.ubcalc.set_u(I)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, places)


def test_i07_horizontal_fails_for_parallel_vectors(hklcalc: HklCalculation):
    hklcalc.ubcalc.set_lattice("Cubic", LatticeParams(1))
    hklcalc.constraints.asdict = {"chi": 0, "phi": 0, "a_eq_b": True}
    hklcalc.ubcalc.set_u(I)

    case = Case("", (0, 0, 1), (30, 0, 60, 90, 0, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case"),
    [
        Case("", (1, 1, 0.001), (-89.9714, 90.0430, -89.9618, 90.0143, 90, 0)),
        Case("", (1, 1, 0.1), (-87.1331, 85.6995, 93.8232, 91.4339, 90, 0)),
        Case("", (1, 1, 0.5), (-75.3995, 68.0801, 109.5630, 97.3603, 90, 0)),
        Case("", (1, 1, 1), (-58.6003, 42.7342, 132.9005, 106.3250, 90, 0)),
    ],
)
def test_i16_vertical(hklcalc: HklCalculation, case: Case):
    hklcalc.ubcalc.set_lattice("Cubic", LatticeParams(1))
    hklcalc.constraints.asdict = {"chi": pi / 2, "psi": pi / 2, "phi": 0}
    hklcalc.ubcalc.set_u(I)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


@pytest.mark.parametrize(
    ("case", "constraints"),
    itertools.product(
        [
            Case(
                "",
                (-1.1812112493619709, -0.71251524866987204, 5.1997083010199221),
                (26, 0, 52, 0, 45.2453, 186.6933 - 360),
                12.39842 / 8,
            )
        ],
        [
            {"delta": 0, "a_eq_b": True, "eta": 0},
            {"delta": 0, "psi": -pi / 2, "eta": 0},
            {"delta": 0, "alpha": Q(17.9776, "deg"), "eta": 0},
        ],
    ),
)
def test_i16_failed_hexagonal_experiment(
    hklcalc: HklCalculation, case: Case, constraints: Dict[str, Union[float, bool]]
):
    hklcalc.ubcalc.set_lattice("I16_test", LatticeParams(4.785, 12.991), "Hexagonal")
    u_matrix = np.array(
        [
            [-9.65616334e-01, -2.59922060e-01, 5.06142415e-03],
            [2.59918682e-01, -9.65629598e-01, -1.32559487e-03],
            [5.23201232e-03, 3.55426382e-05, 9.99986312e-01],
        ]
    )
    hklcalc.ubcalc.set_u(u_matrix)

    hklcalc.constraints.asdict = constraints

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 3)


def test_i16_failed_hexagonal_experiment_with_small_variations(hklcalc: HklCalculation):
    hklcalc.ubcalc.set_lattice("I16_test", LatticeParams(4.785, 12.991), "Hexagonal")
    u_matrix = np.array(
        [
            [-9.65616334e-01, -2.59922060e-01, 5.06142415e-03],
            [2.59918682e-01, -9.65629598e-01, -1.32559487e-03],
            [5.23201232e-03, 3.55426382e-05, 9.99986312e-01],
        ]
    )
    hklcalc.ubcalc.set_u(u_matrix)
    hklcalc.constraints.asdict = {"delta": 0, "alpha": Q(17.8776, "deg"), "eta": 0}

    case = Case(
        "",
        (-1.1812112493619709, -0.71251524866987204, 5.1997083010199221),
        (25.85, 0, 52, 0, 45.2453, -173.518),
        12.39842 / 8,
    )
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 3)
