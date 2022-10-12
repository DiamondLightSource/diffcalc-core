from math import isnan
from typing import Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import I


@pytest.fixture
def cubic() -> HklCalculation:
    constraints = Mock()
    constraints.is_fully_constrained.return_value = True

    ubcalc = UBCalculation()
    ubcalc.set_lattice("xtal", 1)
    ubcalc.set_u(I)
    ubcalc.n_phi = (0, 0, 1)  # type: ignore

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
