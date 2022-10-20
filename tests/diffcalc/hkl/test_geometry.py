from math import pi
from typing import Tuple

import numpy as np
import pytest
from diffcalc.hkl.geometry import Position, get_rotation_matrices, ureg
from diffcalc.util import I
from numpy import array

x = array([[1], [0], [0]])
y = array([[0], [1], [0]])
z = array([[0], [0], [1]])


def test_comparison_between_radian_and_degree_positions():
    pos_in_degrees = Position(*(90, 90, 90, 45, 30, 180) * ureg.deg)
    pos_in_radians = Position(pi / 2, pi / 2, pi / 2, pi / 4, pi / 6, pi)

    assert pos_in_degrees == pos_in_radians


def test_comparison_between_positions():
    pos1 = Position(1, 2, 3, 4, 5, 6)
    pos2 = Position(1, 2, 3, 4, 5, 6)
    pos3 = Position(1, 2, 3, 4, 5, 10)

    pos4 = 4

    assert pos1 == pos2
    assert pos1 != pos3
    assert pos1 != pos4


def test_position_str_shows_degree_values_if_degree_position_given():
    pos = Position(90 * ureg.deg, 0, 0, 45 * ureg.deg, 5 * ureg.deg, 0)

    assert (
        str(pos)
        == "Position(mu: 90.0000, delta: 0.0000, nu: 0.0000, eta: 45.0000, chi: "
        + "5.0000, phi: 0.0000)"
    )


def test_position_str_shows_radian_values_if_radian_position_given():
    mu, delta, nu, eta, chi, phi = pi / 4, pi / 4, pi / 6, pi / 2, pi / 2, pi / 2
    pos = Position(mu, delta, nu, eta, chi, phi)

    assert str(pos) == (
        "Position("
        + f"mu: {mu:.4f}, "
        + f"delta: {delta:.4f}, "
        + f"nu: {nu:.4f}, "
        + f"eta: {eta:.4f}, "
        + f"chi: {chi:.4f}, "
        + f"phi: {phi:.4f})"
    )


def test_position_gets_and_sets_degrees_correctly():
    pos_deg = Position()
    angles = {"mu": 90, "delta": 90, "nu": 45, "eta": 30, "chi": 180, "phi": 60}

    for angle, value in angles.items():
        setattr(pos_deg, angle, value * ureg.deg)
        retrieved_value = getattr(pos_deg, angle)
        assert np.round(retrieved_value.magnitude, 8) == np.round(value, 8)


def test_position_gets_and_sets_radians_correctly():
    pos_deg = Position()
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


@pytest.mark.parametrize(
    ("position"),
    [
        (0, pi / 2, pi / 4, pi, pi, pi),
        (
            0,
            90 * ureg.deg,
            45 * ureg.deg,
            180 * ureg.deg,
            180 * ureg.deg,
            180 * ureg.deg,
        ),
    ],
)
def test_get_rotation_matrices_returns_correct_matrices(
    position: Tuple[float, float, float, float, float, float]
):
    matrices = get_rotation_matrices(Position(*position))

    assert np.all(matrices[0] == I)
    assert np.all(np.abs(np.round(matrices[3])) == I)
    assert np.all(np.abs(np.round(matrices[4])) == I)
    assert np.all(np.abs(np.round(matrices[5])) == I)
