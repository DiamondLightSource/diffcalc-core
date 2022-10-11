from dataclasses import dataclass
from math import radians
from typing import Dict, Tuple

import numpy as np
import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import I, y_rotation, z_rotation

from tests.tools import (
    assert_array_almost_equal_in_list,
    assert_dict_almost_in_list,
    assert_second_dict_almost_in_first,
)


@dataclass
class Case:
    name: str
    hkl: Tuple[float, float, float]
    position: Tuple[float, float, float, float, float, float]
    wavelength: float = 1


@pytest.fixture
def cubic() -> HklCalculation:
    ubcalc = UBCalculation("test_str")
    ubcalc.n_phi = (0, 0, 1)
    ubcalc.surf_nphi = (0, 0, 1)
    ubcalc.set_lattice("xtal", "Cubic", 1)
    ubcalc.add_reflection((0, 0, 1), Position(0, 60, 0, 30, 0, 0), 12.4, "ref1")
    ubcalc.add_orientation((0, 1, 0), (0, 1, 0), Position(1, 0, 0, 0, 2, 0), "orient1")
    ubcalc.set_u(I)

    constraints = Constraints()
    constraints.nu = 0
    constraints.psi = 90
    constraints.phi = 90

    return HklCalculation(ubcalc, constraints)


def test_str(cubic):
    assert (
        str(cubic)
        == """    DET             REF             SAMP
    -----------     -----------     -----------
    delta           a_eq_b          mu
--> nu              alpha           eta
    qaz             beta            chi
    naz         --> psi         --> phi
                    bin_eq_bout     bisect
                    betain          omega
                    betaout

    nu   : 0.0000
    psi  : 90.0000
    phi  : 90.0000
"""
    )


def test_serialisation(cubic: HklCalculation):
    hkl_json = cubic.asdict

    hklcalc = HklCalculation.fromdict(hkl_json)

    assert hklcalc.asdict == hkl_json


def configure_ub(hkl: HklCalculation, zrot: float, yrot: float) -> None:
    ZROT = z_rotation(radians(zrot))  # -PHI
    YROT = y_rotation(radians(yrot))  # +CHI
    U = ZROT @ YROT
    hkl.ubcalc.set_u(np.array(U))


def convert_position_to_hkl_and_hkl_to_position(
    hklcalc: HklCalculation,
    case: Case,
    places: int = 5,
    expected_virtual: Dict[str, float] = {},
) -> None:

    position: Position = Position(*case.position)
    hkl = hklcalc.get_hkl(position, case.wavelength)

    assert np.all(np.round(hkl, places) == np.round(case.hkl, places))

    pos_virtual_angles_pairs_in_degrees = hklcalc.get_position(
        case.hkl[0], case.hkl[1], case.hkl[2], case.wavelength
    )

    pos = [result[0] for result in pos_virtual_angles_pairs_in_degrees]
    virtual_from_get_position = [
        result[1] for result in pos_virtual_angles_pairs_in_degrees
    ]

    assert_array_almost_equal_in_list(
        position.astuple,
        [p.astuple for p in pos],
        places,
    )

    if expected_virtual:
        virtual_angles = hklcalc.get_virtual_angles(position)
        assert_second_dict_almost_in_first(virtual_angles, expected_virtual)
        assert_dict_almost_in_list(virtual_from_get_position, expected_virtual)
