from math import atan, pi, sqrt

import numpy as np
import pytest
from diffcalc.ub.crystal import Crystal, LatticeParams
from diffcalc.util import DiffcalcException

from tests.diffcalc import Q, scenarios


def test_correct_b_matrix_from_init():
    for scenario in scenarios.sessions():
        lattice = LatticeParams(*scenario.lattice)
        cut = Crystal("tc", lattice)

        assert np.all(np.round(cut.B, 4) == np.round(scenario.bmatrix, 4)), (
            "Incorrect B matrix calculation for scenario " + scenario.name
        )


@pytest.mark.parametrize(
    ("xtal_system", "unit_cell", "full_unit_cell"),
    [
        (
            "Triclinic",
            (1, 2, 3, Q(10, "deg"), Q(20, "deg"), Q(30, "deg")),
            (1, 2, 3, Q(10, "deg"), Q(20, "deg"), Q(30, "deg")),
        ),
        (
            "Monoclinic",
            (1, 2, 3, Q(20, "deg")),
            (1, 2, 3, Q(90, "deg"), Q(20, "deg"), Q(90, "deg")),
        ),
        (
            "Orthorhombic",
            (1, 2, 3),
            (1, 2, 3, Q(90, "deg"), Q(90, "deg"), Q(90, "deg")),
        ),
        ("Tetragonal", (1, 3), (1, 1, 3, Q(90, "deg"), Q(90, "deg"), Q(90, "deg"))),
        (
            "Rhombohedral",
            (1, Q(10, "deg")),
            (1, 1, 1, Q(10, "deg"), Q(10, "deg"), Q(10, "deg")),
        ),
        ("Cubic", (1,), (1, 1, 1, Q(90, "deg"), Q(90, "deg"), Q(90, "deg"))),
        pytest.param(
            "Orthorombic",
            (1, 2, 3),
            (1, 2, 3, Q(90, "deg"), Q(90, "deg"), Q(90, "deg")),
            marks=pytest.mark.xfail(raises=DiffcalcException),
        ),
    ],
)
def test_get_lattice_and_get_lattice_params(xtal_system, unit_cell, full_unit_cell):
    lattice = LatticeParams(*unit_cell)
    xtal = Crystal("xtal", lattice, xtal_system)

    assert unit_cell == xtal.get_lattice_params()
    assert xtal_system == xtal.system

    assert xtal.get_lattice() == full_unit_cell


@pytest.mark.parametrize(
    ("hkl1", "hkl2", "angle"),
    [
        ([0, 0, 1], [0, 0, 2], 0),
        ([0, 1, 0], [0, 0, 2], pi / 2),
        ([1, 0, 1], [0, 0, 2], pi / 4),
        ([1, 1, 1], [0, 0, 2], atan(sqrt(2.0))),
    ],
)
def test_get_hkl_plane_angle(hkl1, hkl2, angle):
    xtal = Crystal("cube", LatticeParams(1, 1, 1, pi / 2, pi / 2, pi / 2))

    assert xtal.get_hkl_plane_angle(hkl1, hkl2) == pytest.approx(angle)


def test_string():
    cut = Crystal("HCl", LatticeParams(1, 2, 3, Q(4, "deg"), Q(5, "deg"), Q(6, "deg")))

    with open(f"tests/diffcalc/ub/strings/crystal/crystal_cut.txt") as f:
        expected_string = f.read()

    assert str(cut) == expected_string


def test_serialisation():
    for sess in scenarios.sessions():
        crystal = Crystal("tc", LatticeParams(*sess.lattice))
        cut_json = crystal.asdict
        reformed_crystal = Crystal(**cut_json)

        assert (reformed_crystal.B == crystal.B).all()
