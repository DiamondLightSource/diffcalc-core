import itertools
from math import cos, radians, sin
from typing import Dict, Union

import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import DiffcalcException

from tests.diffcalc.hkl.test_calc import (
    Case,
    configure_ub,
    convert_position_to_hkl_and_hkl_to_position,
)


@pytest.fixture
def cubic() -> UBCalculation:
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore

    ubcalc.set_lattice("Cubic", 1.0)
    return ubcalc


@pytest.mark.parametrize(
    ("case"),
    (
        Case("101", (1, 0, 1), (45, 0, 90, 0, 45, 180)),
        Case("10m1", (1, 0, -1), (45, 0, 90, 0, 135, 180)),
        Case("011", (0, 1, 1), (45, 0, 90, 0, 45, 270)),
        Case(
            "100-->001", (sin(radians(4)), 0, cos(radians(4))), (30, 0, 60, 0, 4, 180)
        ),
        Case("010", (0, 1, 0), (30, 0, 60, 0, 90, 270)),
        Case("0.1 0 1.5", (0.1, 0, 1.5), (48.7348, 0, 97.46959, 0, 3.81407, 180)),
        Case(
            "010-->001", (0, cos(radians(86)), sin(radians(86))), (30, 0, 60, 0, 4, 270)
        ),
    ),
)
def test_bisect(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(
        cubic, Constraints({"delta": 0, "bisect": True, "omega": 0})
    )

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


def test_bisect_fails_for_non_unique_phi(cubic: UBCalculation):
    hklcalc = HklCalculation(
        cubic, Constraints({"delta": 0, "bisect": True, "omega": 0})
    )

    case = Case("001", (0, 0, 1), (30, 0, 60, 0, 0, 0))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case", "constraints"),
    itertools.product(
        [
            Case(
                "100-->001",
                (cos(radians(4)), 0, sin(radians(4))),
                (30, 0, 60, 0, 90 - 4, -180),
            ),
            Case(
                "001-->100",
                (cos(radians(86)), 0, sin(radians(86))),
                (30, 0, 60, 0, -4, 0),
            ),
        ],
        [
            {"a_eq_b": True, "qaz": 0, "eta": 0},
            {"a_eq_b": True, "delta": 0, "eta": 0},
        ],
    ),
)
def test_various_constraints(
    cubic: UBCalculation,
    case: Case,
    constraints: Dict[str, Union[float, bool]],
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case", "constraints"),
    itertools.product(
        [
            Case("100", (1, 0, 0), (30, 0, 60, 0, 90, -180)),
            Case("010", (0, 1, 0), (30, 0, 60, 0, 90, -90)),
            Case("001", (0, 0, 1), (30, 0, 60, 0, 0, 0)),
        ],
        [
            {"a_eq_b": True, "qaz": 0, "eta": 0},
            {"a_eq_b": True, "delta": 0, "eta": 0},
        ],
    ),
)
def test_fails_when_a_eq_b_and_parallel_vectors(
    cubic: UBCalculation,
    case: Case,
    constraints: Dict[str, Union[float, bool]],
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case", "yrot", "constraints"),
    itertools.product(
        [
            Case("100", (1, 0, 0), (30, 0, 60, 0, 90, -180)),
            Case(
                "100-->001",
                (cos(radians(4)), 0, sin(radians(4))),
                (30, 0, 60, 0, 90 - 4, -180),
            ),
            Case("001", (0, 0, 1), (30, 0, 60, 0, 0, 0)),
            Case(
                "001-->100",
                (cos(radians(86)), 0, sin(radians(86))),
                (30, 0, 60, 0, -4, 0),
            ),
        ],
        [2, -2],
        [
            {"a_eq_b": True, "qaz": 0, "eta": 0},
            {"a_eq_b": True, "delta": 0, "eta": 0},
        ],
    ),
)
def test_various_constraints_and_yrots(
    cubic: UBCalculation,
    case: Case,
    yrot: float,
    constraints: Dict[str, Union[float, bool]],
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))
    zrot = 1

    if case.name.startswith("100"):
        case_with_rotation = Case(
            case.name,
            case.hkl,
            (*case.position[:4], case.position[4] + yrot, case.position[5] + zrot),
        )
    else:
        case_with_rotation = Case(
            case.name,
            case.hkl,
            (*case.position[:4], case.position[4] - yrot, case.position[5] + zrot),
        )

    configure_ub(hklcalc, zrot, yrot)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


@pytest.mark.parametrize(
    ("yrot", "constraints"),
    [
        (2, {"a_eq_b": True, "qaz": 0, "eta": 0}),
        (2, {"a_eq_b": True, "delta": 0, "eta": 0}),
        (-2, {"a_eq_b": True, "qaz": 0, "eta": 0}),
        (-2, {"a_eq_b": True, "delta": 0, "eta": 0}),
    ],
)
def test_one_parallel_vector_still_fails_even_if_rotated(
    cubic: UBCalculation, yrot: float, constraints: Dict[str, Union[bool, float]]
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    case = Case("010", (0, 1, 0), (30, 0, 60, 0, 90, -89))

    configure_ub(hklcalc, 1, yrot)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)
