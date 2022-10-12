import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation

from tests.diffcalc.hkl.test_calc import (
    Case,
    convert_position_to_hkl_and_hkl_to_position,
)


@pytest.fixture
def tetragonal() -> UBCalculation:
    ubcalc = UBCalculation()

    ubcalc.set_lattice(name="test", a=4.913, c=5.405)
    ubcalc.add_reflection(
        (0, 0, 1), Position(7.31, 0, 10.62, 0, 0, 0), 12.39842, "refl1"
    ),
    ubcalc.add_orientation((0, 1, 0), (0, 1, 0), tag="plane")

    ubcalc.n_hkl = (1.0, 0, 0)  # type: ignore

    return ubcalc


def calculate_ub(hkl: HklCalculation):
    hkl.ubcalc.calc_ub("refl1", "plane")


@pytest.mark.parametrize(
    ("case"),
    (
        Case("", (1, 1, 1), (90.8684, -15.0274, 12.8921, 1, -29.5499, -87.7027)),
        Case("", (1, 2, 3), (90.8265, -39.0497, 17.0693, 1, -30.4506, -87.6817)),
    ),
)
def test_naz_alpha_eta(tetragonal: UBCalculation, case: Case):
    hklcalc = HklCalculation(
        tetragonal, Constraints({"naz": 3.0, "alpha": 2.0, "eta": 1.0})
    )

    calculate_ub(hklcalc)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)
