# tests the UBCalculation object.

from math import radians, sqrt
from pathlib import Path

import numpy as np
import pytest
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import DiffcalcException

from tests.diffcalc import scenarios

# Integration tests


@pytest.mark.integtest
class TestStrings:
    def create(self, name: str) -> None:
        self.ubcalc = UBCalculation(name)

    def retrieve_expected_string(self, name: str) -> str:
        with open(f"tests/diffcalc/ub/strings/{name}.txt", "r") as f:
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


@pytest.mark.integtest
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


# unit tests


@pytest.fixture
def ubcalc():
    return UBCalculation("testing_set_lattice")


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


def test_setu_and_setub():
    ubcalc = UBCalculation("test_setu_and_setub")
    with pytest.raises(TypeError):
        ubcalc.set_u([[1, 2], [3, 4]])

    ubcalc.set_u([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    with pytest.raises(TypeError):
        ubcalc.set_ub([[1, 2], [3, 4]])

    ubcalc.set_ub([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_calc_ub():
    ubcalc = UBCalculation("test_calc_ub")
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


def test_calc_ub_for_mixture_of_orientations_and_reflections():
    ubcalc = UBCalculation("test_variety_calc_ub")
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


def test_refine_ub():
    ubcalc = UBCalculation("testing_refine_ub")
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
