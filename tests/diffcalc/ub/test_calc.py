"""Test the UBCalculation and ReferenceVector objects."""

from math import degrees, radians, sqrt
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import ReferenceVector, UBCalculation
from diffcalc.util import DiffcalcException
from numpy import array

from tests.diffcalc import scenarios

# Integration tests


class TestStrings:
    def create(self, name: str) -> None:
        self.ubcalc = UBCalculation(name)

    def retrieve_expected_string(self, name: str) -> str:
        with open(f"tests/diffcalc/ub/strings/calc/{name}.txt") as f:
            expected_string = f.read()

        return expected_string

    def test_str_n_phi(self):
        self.create("test_str_none")

        self.ubcalc.n_phi = (0, 0, 1)
        self.ubcalc.surf_nphi = (0, 0, 1)

        assert str(self.ubcalc) == self.retrieve_expected_string("n_phi")

    def test_str_n_hkl(self):
        self.create("test_str_none")

        self.ubcalc.n_hkl = (0, 0, 1)
        self.ubcalc.surf_nhkl = (0, 0, 1)

        assert str(self.ubcalc) == self.retrieve_expected_string("n_hkl")

    def test_str_UB_unity(self):
        self.create("test_str_UB_unity")

        self.ubcalc.n_phi = (0, 0, 1)
        self.ubcalc.surf_nphi = (0, 0, 1)
        self.ubcalc.set_lattice("xtal", "Cubic", 1)
        self.ubcalc.set_miscut(None, 0.0)

        assert str(self.ubcalc) == self.retrieve_expected_string("unitary_UB")

    def test_str_refl_orient_UB(self):
        self.create("test_str")

        self.ubcalc.n_phi = (0, 0, 1)
        self.ubcalc.surf_nphi = (0, 0, 1)
        self.ubcalc.set_lattice("xtal", "Cubic", 1)
        self.ubcalc.add_reflection(
            (0, 0, 1), Position(0, 60, 0, 30, 0, 0), 12.4, "ref1"
        )
        self.ubcalc.add_orientation(
            (0, 1, 0), (0, 1, 0), Position(1, 0, 0, 0, 2, 0), "orient1"
        )
        self.ubcalc.set_miscut(None, radians(2.0))

        assert str(self.ubcalc) == self.retrieve_expected_string("full_info")


class TestPersistenceMethods:
    ubcalc = UBCalculation("test_persistence")
    ubcalc.n_phi = (0, 0, 1)
    ubcalc.surf_nphi = (0, 0, 1)
    ubcalc.set_lattice("xtal", "Cubic", 1)
    ubcalc.add_reflection((0, 0, 1), Position(0, 60, 0, 30, 0, 0), 12.4, "ref1")
    ubcalc.add_orientation((0, 1, 0), (0, 1, 0), Position(1, 0, 0, 0, 2, 0), "orient1")
    ubcalc.set_miscut(None, radians(2.0))

    def test_pickling(self):
        self.ubcalc.pickle("test_file")
        assert Path("test_file").exists()

    def test_unpickling(self):
        loaded_ub = UBCalculation.load("test_file")

        assert np.all(loaded_ub.UB == self.ubcalc.UB)

        Path("test_file").unlink()

    def test_unpickling_non_existent_file(self):
        with pytest.raises(FileNotFoundError):
            UBCalculation.load("non-existent-file")

    def test_unpickling_invalid_file(self):
        with pytest.raises(DiffcalcException):
            UBCalculation.load("requirements_dev.txt")


# Test Reflections and Orientations


@pytest.fixture
def ubcalc_ref_orient() -> UBCalculation:
    ubcalc = UBCalculation("test")
    ubcalc.add_reflection((0, 1, 0), Position(1, 2, 3, 4, 5, 6), 12, "refl")
    ubcalc.add_orientation((0, 1, 0), (0, 1, 0), Position(0, 0, 0, 0, 0, 0), "orient")
    return ubcalc


def test_edit_and_retrieve_reflection(ubcalc_ref_orient: UBCalculation):
    orig_reflection = ubcalc_ref_orient.get_reflection(1)
    new_hkl = (0, 2, 0)

    ubcalc_ref_orient.edit_reflection(
        1, new_hkl, orig_reflection.pos, orig_reflection.energy
    )
    new_reflection = ubcalc_ref_orient.get_reflection(1)

    assert np.all((new_reflection.h, new_reflection.k, new_reflection.l) == new_hkl)


def test_delete_reflection(ubcalc_ref_orient: UBCalculation):
    ubcalc_ref_orient.del_reflection(1)

    with pytest.raises(IndexError):
        ubcalc_ref_orient.get_reflection(1)


def test_swap_reflection(ubcalc_ref_orient: UBCalculation):
    orig_reflection = ubcalc_ref_orient.get_reflection(1)

    ubcalc_ref_orient.add_reflection((0, 1, 0), Position(1, 2, 3, 4, 5, 6), 13, "refl1")
    ubcalc_ref_orient.swap_reflections(1, 2)

    assert ubcalc_ref_orient.get_reflection(2) == orig_reflection


def test_edit_and_retrieve_orientation(ubcalc_ref_orient: UBCalculation):
    orig_orientation = ubcalc_ref_orient.get_orientation(1)
    orig_xyz = [orig_orientation.x, orig_orientation.y, orig_orientation.z]

    new_hkl = (0, 2, 0)
    ubcalc_ref_orient.edit_orientation(1, new_hkl, orig_xyz, orig_orientation.pos)
    new_orientation = ubcalc_ref_orient.get_orientation(1)

    assert np.all((new_orientation.h, new_orientation.k, new_orientation.l) == new_hkl)


def test_delete_orientation(ubcalc_ref_orient: UBCalculation):
    ubcalc_ref_orient.del_orientation(1)

    with pytest.raises(IndexError):
        ubcalc_ref_orient.get_orientation(1)


def test_swap_orientation(ubcalc_ref_orient: UBCalculation):
    orig_orientation = ubcalc_ref_orient.get_orientation(1)

    ubcalc_ref_orient.add_orientation(
        (0, 1, 0), (0, 3, 0), Position(1, 2, 3, 4, 5, 6), "orient1"
    )
    ubcalc_ref_orient.swap_orientations(1, 2)

    assert ubcalc_ref_orient.get_orientation(2) == orig_orientation


def test_get_number_reflections_and_orientations_gets_correct_value(
    ubcalc_ref_orient: UBCalculation,
):
    assert ubcalc_ref_orient.get_number_reflections() == 1
    assert ubcalc_ref_orient.get_number_orientations() == 1


def test_get_tag_gets_correct_value(ubcalc_ref_orient: UBCalculation):
    assert ubcalc_ref_orient.get_tag_refl_num("refl") == 1
    assert ubcalc_ref_orient.get_tag_orient_num("orient") == 1


# unit tests


@pytest.fixture
def ubcalc():
    return UBCalculation("test")


def test_set_lattice_one_argument_guesses_cubic(ubcalc):
    ubcalc.set_lattice("NaCl", 1.1)

    assert ("NaCl", 1.1, 1.1, 1.1, 90, 90, 90) == ubcalc.crystal.get_lattice()

    assert ubcalc.crystal.system == "Cubic"


def test_set_lattice_two_arguments_guesses_tetragonal(ubcalc):
    ubcalc.set_lattice("NaCl", 1.1, 2.2)

    assert ("NaCl", 1.1, 1.1, 2.2, 90, 90, 90) == ubcalc.crystal.get_lattice()

    assert ubcalc.crystal.system == "Tetragonal"


def test_set_lattice_three_arguments_guesses_orthorhombic(ubcalc):
    ubcalc.set_lattice("NaCl", 1.1, 2.2, 3.3)

    assert ("NaCl", 1.1, 2.2, 3.3, 90, 90, 90) == ubcalc.crystal.get_lattice()

    assert ubcalc.crystal.system == "Orthorhombic"


def test_set_lattice_four_arguments_guesses_monoclinic(ubcalc):
    ubcalc.set_lattice("NaCl", 1.1, 2.2, 3.3, 91)
    assert ("NaCl", 1.1, 2.2, 3.3, 90, 91, 90) == ubcalc.crystal.get_lattice()

    assert ubcalc.crystal.system == "Monoclinic"


def test_set_lattice_five_arguments_raises_error(ubcalc):
    with pytest.raises(TypeError):
        ubcalc.set_lattice(("NaCl", 1.1, 2.2, 3.3, 91, 92))


def test_set_lattice_six_arguments_guesses_triclinic(ubcalc):
    ubcalc.set_lattice("NaCl", 1.1, 2.2, 3.3, 91, 92, 93)

    assert (
        "NaCl",
        1.1,
        2.2,
        3.3,
        91,
        92,
        93,
    ) == ubcalc.crystal.get_lattice()

    assert ubcalc.crystal.system == "Triclinic"


def test_setu_and_setub(ubcalc):
    with pytest.raises(TypeError):
        ubcalc.set_u([[1, 2], [3, 4]])

    ubcalc.set_u([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    with pytest.raises(TypeError):
        ubcalc.set_ub([[1, 2], [3, 4]])

    ubcalc.set_ub([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_calc_ub(ubcalc):
    with pytest.raises(DiffcalcException):
        ubcalc.calc_ub()

    for scenario in scenarios.sessions():
        ubcalc = UBCalculation("test_calcub")
        ubcalc.set_lattice(scenario.name, *scenario.lattice)

        for ref in [scenario.ref1, scenario.ref2]:
            ubcalc.add_reflection((ref.h, ref.k, ref.l), ref.pos, ref.energy, ref.tag)

        ubcalc.calc_ub(scenario.ref1.tag, scenario.ref2.tag)

        assert np.all(
            np.round(ubcalc.UB, 4)
            == np.round(np.array(scenario.umatrix) @ np.array(scenario.bmatrix), 4)
        ), "wrong UB matrix after calculating U"


def test_calc_ub_for_mixture_of_orientations_and_reflections(ubcalc):
    scenario = scenarios.sessions()[1]
    ubcalc.set_lattice(scenario.name, *scenario.lattice)

    for idx, orient in enumerate([scenario.ref1, scenario.ref2]):
        xyz = list(scenario.umatrix[idx])
        hkl = [orient.h, orient.k, orient.l]
        ubcalc.add_orientation(hkl, xyz, tag=f"or{idx+1}")

    ubcalc.calc_ub()

    assert np.all(
        np.round(ubcalc.UB, 4)
        == np.round(np.array(scenario.umatrix) @ np.array(scenario.bmatrix), 4)
    ), "wrong UB matrix after calculating U"

    ubcalc.calc_ub("or1", "or2")
    assert np.all(
        np.round(ubcalc.UB, 4)
        == np.round(np.array(scenario.umatrix) @ np.array(scenario.bmatrix), 4)
    ), "wrong UB matrix after calculating U"

    for ref in [scenario.ref1, scenario.ref2]:
        ubcalc.add_reflection((ref.h, ref.k, ref.l), ref.pos, ref.energy, ref.tag)

    ubcalc.calc_ub("or1", scenario.ref2.tag)
    assert np.all(
        np.round(ubcalc.UB, 4)
        == np.round(np.array(scenario.umatrix) @ np.array(scenario.bmatrix), 4)
    ), "wrong UB matrix after calculating U"


def test_refine_ub(ubcalc):
    ubcalc.set_lattice("xtal", 1, 1, 1, 90, 90, 90)
    ubcalc.set_miscut(None, 0)
    ubcalc.refine_ub(
        (1, 1, 0),
        Position(0, 60, 0, 30, 0, 0),
        1.0,
        True,
        True,
    )
    assert ("xtal", sqrt(2.0), sqrt(2.0), 1, 90, 90, 90) == ubcalc.crystal.get_lattice()

    assert np.all(
        np.round(ubcalc.U, 4)
        == np.array(
            [
                [0.7071, 0.7071, 0],
                [-0.7071, 0.7071, 0],
                [0, 0, 1],
            ]
        )
    ), "wrong U matrix after refinement"


def test_fit_ub():
    for scenario in scenarios.sessions()[-3:]:
        ubcalc = UBCalculation("test_fit_ub_matrix")
        a, b, c, alpha, beta, gamma = scenario.lattice

        for ref in scenario.reflist:
            ubcalc.add_reflection((ref.h, ref.k, ref.l), ref.pos, ref.energy, ref.tag)

        ubcalc.set_lattice(scenario.name, scenario.system, *scenario.lattice)
        ubcalc.calc_ub(scenario.ref1.tag, scenario.ref2.tag)

        init_latt = (
            1.06 * a,
            1.07 * b,
            0.94 * c,
            1.05 * alpha,
            1.06 * beta,
            0.95 * gamma,
        )
        ubcalc.set_lattice(scenario.name, scenario.system, *init_latt)
        ubcalc.set_miscut([0.2, 0.8, 0.1], 3.0, True)

        ubcalc.fit_ub([r.tag for r in scenario.reflist], True, True)

        assert np.all(
            np.round([ubcalc.crystal.get_lattice()[1:]], 2)
            == np.round(scenario.lattice, 2)
        ), "wrong lattice after fitting UB"

        assert np.all(
            np.round(ubcalc.U, 3) == np.round(scenario.umatrix, 3)
        ), "wrong U matrix after fitting UB"


def test_get_ttheta_from_hkl(ubcalc):
    ubcalc.set_lattice("cube", 1, 1, 1, 90, 90, 90)
    assert ubcalc.get_ttheta_from_hkl((0, 0, 1), 12.39842) == pytest.approx(radians(60))


@pytest.mark.parametrize(
    ("miscut_xyzs", "miscut_angles", "add_miscuts", "expected_n_hkls"),
    [
        (
            [(0, 1, 0), (0, 1, 0), (0, -1, 0)],
            [np.pi / 6, np.pi / 12, np.pi / 4],
            [False, True, True],
            [
                array([[-0.5], [0], [0.8660254]]),
                array([[-0.7071068], [0], [0.7071068]]),
                array([[0.0], [0.0], [1.0]]),
            ],
        ),
    ],
)
def test_set_miscut(
    miscut_xyzs: List[Tuple[float, float, float]],
    miscut_angles: List[float],
    add_miscuts: List[bool],
    expected_n_hkls: List[np.ndarray],
):
    ubcalc = UBCalculation("testsetmiscut")
    ubcalc.reference = ReferenceVector((0, 0, 1), False)
    ubcalc.set_lattice("cube", 1, 1, 1, 90, 90, 90)

    for idx in range(len(miscut_xyzs)):
        ubcalc.set_miscut(miscut_xyzs[idx], miscut_angles[idx], add_miscuts[idx])
        assert np.all(np.round(ubcalc.n_hkl, 5) == np.round(expected_n_hkls[idx], 5))


@pytest.mark.parametrize(
    ("axis", "angle", "hkl", "pos"),
    [
        ((0, 1, 0), 10, (0, 0, 1), Position(40, 0, 60, 0, 0, -90)),
        ((1, 0, 0), 30, (0, 1, 1), Position(45, 0, 90, 0, 15, -90)),
    ],
)
def test_get_miscut_from_hkl(
    ubcalc: UBCalculation,
    axis: Tuple[float, float, float],
    angle: float,
    hkl: Tuple[float, float, float],
    pos: Position,
):
    ubcalc.set_lattice("xtal", 1, 1, 1, 90, 90, 90)
    ubcalc.set_miscut(axis, radians(angle))
    ubcalc.set_miscut(None, 0)

    miscut, miscut_axis = ubcalc.get_miscut_from_hkl(hkl, pos)

    assert miscut == pytest.approx(angle)
    assert np.all(
        np.round(axis, 4) == np.round(miscut_axis, 4)
    ), "wrong calculation for miscut axis"


@pytest.mark.parametrize(
    ("axis", "angle"),
    [((0, 1, 0), 10), ((1, 0, 0), 30), ((0.70710678, 0.70710678, 0), 50)],
)
def test_get_miscut(ubcalc, axis, angle):
    ubcalc.set_lattice("xtal", 1, 1, 1, 90, 90, 90)
    ubcalc.set_miscut(axis, radians(angle))

    test_angle, test_axis = ubcalc.get_miscut()

    assert degrees(test_angle) == pytest.approx(angle)
    assert np.all(np.round(test_axis.T, 4) == np.round(axis, 4))
