from typing import Tuple

import numpy as np
import pytest
from diffcalc.hkl.geometry import Position, get_rotation_matrices
from diffcalc.util import I

x = np.array([[1], [0], [0]])
y = np.array([[0], [1], [0]])
z = np.array([[0], [0], [1]])


def test_comparison_between_positions():
    pos1 = Position(1, 2, 3, 4, 5, 6)
    pos2 = Position(1, 2, 3, 4, 5, 6)
    pos3 = Position(1, 2, 3, 4, 5, 10)

    pos4 = 4

    assert pos1 == pos2
    assert pos1 != pos3
    assert pos1 != pos4


def test_position_string():
    pos = Position(1, 2, 3, 4, 5, 6)

    assert (
        str(pos)
        == "Position(mu: 1.0000, delta: 2.0000, nu: 3.0000, eta: 4.0000, chi: 5.0000, phi: 6.0000)"
    )


def test_position_gets_and_sets_degrees_correctly():
    pos_deg = Position()
    angles = {"mu": 90, "delta": 90, "nu": 45, "eta": 30, "chi": 180, "phi": 60}

    for angle, value in angles.items():
        setattr(pos_deg, angle, value)
        retrieved_value = getattr(pos_deg, angle)
        assert np.round(retrieved_value, 8) == np.round(value, 8)


@pytest.mark.parametrize(("position"), [(0, 90, 45, 180, 180, 180)])
def test_get_rotation_matrices_returns_correct_matrices(
    position: Tuple[float, float, float, float, float, float]
):
    matrices = get_rotation_matrices(Position(*position))

    assert np.allclose(matrices[0], I)
    assert np.allclose(np.abs(matrices[3]), I)
    assert np.allclose(np.abs(matrices[4]), I)
    assert np.allclose(np.abs(matrices[5]), I)


def test_delete_position_properties():
    position = Position(1, 2, 3, 4, 5, 6)

    del position.mu
    del position.delta
    del position.nu
    del position.eta
    del position.chi
    del position.phi

    assert np.isnan(position.mu)
    assert np.isnan(position.delta)
    assert np.isnan(position.nu)
    assert np.isnan(position.eta)
    assert np.isnan(position.chi)
    assert np.isnan(position.phi)

    assert all(np.isnan(v) for v in position.asdict.values())
