from math import degrees, isnan, pi
from typing import Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import I
from numpy import array

x = array([[1], [0], [0]])
y = array([[0], [1], [0]])
z = array([[0], [0], [1]])


def test_comparison_between_radian_and_degree_positions():
    pos_in_degrees = Position(90, 90, 90, 45, 30, 180)
    pos_in_radians = Position(
        pi / 2, pi / 2, pi / 2, pi / 4, pi / 6, pi, indegrees=False
    )

    assert pos_in_degrees == pos_in_radians


def test_comparison_between_positions():
    pos1 = Position(1, 2, 3, 4, 5, 6)
    pos2 = Position(1, 2, 3, 4, 5, 6)
    pos3 = Position(1, 2, 3, 4, 5, 10)

    assert pos1 == pos2
    assert pos1 != pos3


def test_position_string():
    pos = Position(1, 2, 3, 4, 5, 6)

    assert (
        str(pos)
        == "Position(mu: 1.0000, delta: 2.0000, nu: 3.0000, eta: 4.0000, chi: 5.0000, phi: 6.0000)"
    )


def test_position_str_converts_radians_to_degrees():
    mu, delta, nu, eta, chi, phi = pi / 4, pi / 4, pi / 6, pi / 2, pi / 2, pi / 2
    pos = Position(mu, delta, nu, eta, chi, phi, indegrees=False)

    assert str(pos) == (
        "Position("
        + f"mu: {degrees(mu):.4f}, "
        + f"delta: {degrees(delta):.4f}, "
        + f"nu: {degrees(nu):.4f}, "
        + f"eta: {degrees(eta):.4f}, "
        + f"chi: {degrees(chi):.4f}, "
        + f"phi: {degrees(phi):.4f})"
    )


def test_position_gets_and_sets_degrees_correctly():
    pos_deg = Position()
    angles = {"mu": 90, "delta": 90, "nu": 45, "eta": 30, "chi": 180, "phi": 60}

    for angle, value in angles.items():
        setattr(pos_deg, angle, value)
        retrieved_value = getattr(pos_deg, angle)
        assert np.round(retrieved_value, 8) == np.round(value, 8)


def test_position_gets_and_sets_radians_correctly():
    pos_deg = Position(indegrees=False)
    angles = {
        "mu": pi,
        "delta": pi,
        "nu": pi / 2,
        "eta": pi / 6,
        "chi": pi,
        "phi": pi / 3,
    }

    for angle, value in angles.items():
        setattr(pos_deg, angle, value)
        retrieved_value = getattr(pos_deg, angle)
        assert np.round(retrieved_value, 8) == np.round(value, 8)


@pytest.fixture
def cubic() -> HklCalculation:
    constraints = Mock()
    constraints.is_fully_constrained.return_value = True

    ubcalc = UBCalculation()
    ubcalc.set_lattice("xtal", 1)
    ubcalc.set_u(I)
    ubcalc.n_phi = (0, 0, 1)

    return HklCalculation(ubcalc, constraints)


@pytest.mark.parametrize(
    ("angle", "expected_values_for_positions"),
    [
        (
            "theta",
            {
                0: [(-99, 0, 0, 99, 99, 99)],
                1: [
                    (-99, 2, 0, 99, 99, 99),
                    (-99, 0, 2, 99, 99, 99),
                    (-99, -2, 0, 99, 99, 99),
                    (-99, 0, -2, 99, 99, 99),
                ],
            },
        ),
        (
            "qaz",
            {
                0: [(-99, 0, 0, 99, 99, 99), (-99, 0, 1, 99, 99, 99)],
                90: [(-99, 2, 0, 99, 99, 99), (-99, 90, 0, 99, 99, 99)],
            },
        ),
        (
            "alpha",
            {
                0: [
                    (0, 99, 99, 0, 0, 0),
                    (0, 99, 99, 0, 0, 10),
                    (0, 99, 99, 0, 0, -10),
                ],
                2: [(2, 99, 99, 0, 0, 0), (0, 99, 99, 90, 2, 0)],
                -2: [(-2, 99, 99, 0, 0, 0)],
            },
        ),
        (
            "beta",
            {
                0: [(0, 10, 0, 6, 0, 5)],
                10: [(0, 0, 10, 0, 0, 0)],
                -10: [(0, 0, -10, 0, 0, 0)],
                5: [(5, 0, 10, 0, 0, 0)],
            },
        ),
        (
            "naz",
            {
                0: [
                    (0, 99, 99, 0, 0, 0),
                    (0, 99, 99, 0, 0, 10),
                    (10, 99, 99, 0, 0, 10),
                ],
                2: [(0, 99, 99, 0, 2, 0)],
                -2: [(0, 99, 99, 0, -2, 0)],
            },
        ),
        (
            "tau",
            {
                90: [(0, 20, 0, 10, 0, 0), (0, 20, 0, 10, 0, 3)],
                88: [(0, 20, 0, 10, 2, 0)],
                92: [(0, 20, 0, 10, -2, 0)],
                10: [(0, 0, 20, 0, 0, 0)],
            },
        ),
        (
            "psi",
            {
                90: [
                    (0, 11, 0, 0, 0, 0),
                    (0, 11, 0, 0, 0, 12.3),
                    (0, 0.001, 0, 90, 0, 0),
                ],
                100: [(10, 0.001, 0, 0, 0, 0)],
                80: [(-10, 0.001, 0, 0, 0, 0)],
                92: [(0, 0.001, 0, 90, 2, 0)],
                88: [(0, 0.001, 0, 90, -2, 0)],
            },
        ),
    ],
)
def test_get_virtual_angles_calculates_correct_angles(
    cubic: HklCalculation,
    angle: str,
    expected_values_for_positions: Dict[
        float, List[Tuple[float, float, float, float, float, float]]
    ],
):
    for expected_value, positions in expected_values_for_positions.items():
        for each_position in positions:
            virtual_angles = cubic.get_virtual_angles(Position(*each_position))
            assert np.round(virtual_angles[angle], 8) == expected_value


@pytest.mark.parametrize(
    ("angle", "positions"),
    [
        ("beta", [(0, 0, 0, 0, 0, 0)]),
        ("tau", [(0, 0, 0, 0, 0, 0)]),
        ("psi", [(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 90, 0)]),
    ],
)
def test_get_virtual_angles_correctly_returns_nan_angles_for_some_positions(
    cubic: HklCalculation,
    angle: str,
    positions: List[Tuple[float, float, float, float, float, float]],
):
    for each_position in positions:
        calculated_value = cubic.get_virtual_angles(Position(*each_position))[angle]
        assert isnan(calculated_value)
