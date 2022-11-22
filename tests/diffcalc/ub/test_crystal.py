from math import atan, degrees, sqrt

import numpy as np
import pytest
from diffcalc.ub.crystal import Crystal

from tests.diffcalc import scenarios


def test_correct_b_matrix_from_init():
    for scenario in scenarios.sessions():
        cut = Crystal("tc", *scenario.lattice)

        assert np.all(np.round(cut.B, 4) == np.round(scenario.bmatrix, 4)), (
            "Incorrect B matrix calculation for scenario " + scenario.name
        )


@pytest.mark.parametrize(
    ("xtal_system", "unit_cell", "full_unit_cell"),
    [
        ("Triclinic", (1, 2, 3, 10, 20, 30), (1, 2, 3, 10, 20, 30)),
        ("Monoclinic", (1, 2, 3, 20), (1, 2, 3, 90, 20, 90)),
        ("Orthorhombic", (1, 2, 3), (1, 2, 3, 90, 90, 90)),
        ("Tetragonal", (1, 3), (1, 1, 3, 90, 90, 90)),
        ("Rhombohedral", (1, 10), (1, 1, 1, 10, 10, 10)),
        ("Cubic", (1,), (1, 1, 1, 90, 90, 90)),
        pytest.param(
            "Orthorombic",
            (1, 2, 3),
            (1, 2, 3, 90, 90, 90),
            marks=pytest.mark.xfail(raises=TypeError),
        ),
    ],
)
def test_get_lattice_and_get_lattice_params(xtal_system, unit_cell, full_unit_cell):
    xtal = Crystal("xtal", xtal_system, *unit_cell)
    test_xtal_system, test_unit_cell = xtal.get_lattice_params()

    assert test_xtal_system == xtal_system
    assert np.all(np.round(test_unit_cell, 5) == np.round(unit_cell, 5))
    assert np.all(np.round(xtal.get_lattice()[1:], 5) == np.round(full_unit_cell, 5))


@pytest.mark.parametrize(
    ("hkl1", "hkl2", "angle"),
    [
        ([0, 0, 1], [0, 0, 2], 0.0),
        ([0, 1, 0], [0, 0, 2], 90.0),
        ([1, 0, 1], [0, 0, 2], 45.0),
        ([1, 1, 1], [0, 0, 2], degrees(atan(sqrt(2.0)))),
    ],
)
def test_get_hkl_plane_angle(hkl1, hkl2, angle):
    xtal = Crystal("cube", 1, 1, 1, 90, 90, 90)

    assert xtal.get_hkl_plane_angle(hkl1, hkl2) == pytest.approx(angle)


def test_string():
    cut = Crystal("HCl", 1, 2, 3, 4, 5, 6)

    with open(f"tests/diffcalc/ub/strings/crystal/crystal_cut.txt") as f:
        expected_string = f.read()

    assert str(cut) == expected_string


def test_serialisation():
    for sess in scenarios.sessions():
        crystal = Crystal("tc", *sess.lattice)
        cut_json = crystal.asdict
        reformed_crystal = Crystal(**cut_json)

        assert (reformed_crystal.B == crystal.B).all()
