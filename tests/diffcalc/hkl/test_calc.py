from dataclasses import dataclass
from math import isnan, pi, radians, sqrt
from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import Mock

import numpy as np
import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import DiffcalcException, I, y_rotation, z_rotation

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
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore

    ubcalc.set_lattice("Cubic", 1.0)
    configure_ub(ubcalc)

    return HklCalculation(ubcalc, Constraints())


@pytest.fixture
def cubic_ub() -> UBCalculation:
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore

    ubcalc.set_lattice("Cubic", 1.0)
    return ubcalc


@pytest.fixture
def hkl_mocked_constraints() -> HklCalculation:
    constraints = Mock()
    constraints.is_fully_constrained.return_value = True

    ubcalc = UBCalculation()
    ubcalc.set_lattice("xtal", 1)
    ubcalc.set_u(I)
    ubcalc.n_phi = (0, 0, 1)  # type: ignore

    return HklCalculation(ubcalc, constraints)


def configure_ub(ubcalc: UBCalculation, zrot: float = 0.0, yrot: float = 0.0) -> None:
    ZROT = z_rotation(radians(zrot))  # -PHI
    YROT = y_rotation(radians(yrot))  # +CHI
    U = ZROT @ YROT
    ubcalc.set_u(np.array(U))


def convert_position_to_hkl_and_hkl_to_position(
    hklcalc: HklCalculation,
    case: Case,
    places: int = 5,
    expected_virtual: Dict[str, float] = {},
    asdegrees: bool = True,
) -> None:

    position: Position = Position(*case.position, indegrees=asdegrees)
    hkl = hklcalc.get_hkl(position, case.wavelength)

    assert np.all(np.round(hkl, places) == np.round(case.hkl, places))

    pos_virtual_angles_pairs_in_degrees = hklcalc.get_position(
        case.hkl[0], case.hkl[1], case.hkl[2], case.wavelength, asdegrees=asdegrees
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


def test_str():
    ubcalc = UBCalculation("test_str")
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore
    ubcalc.set_lattice("xtal", "Cubic", 1)
    ubcalc.add_reflection((0, 0, 1), Position(0, 60, 0, 30, 0, 0), 12.4, "ref1")
    ubcalc.add_orientation((0, 1, 0), (0, 1, 0), Position(1, 0, 0, 0, 2, 0), "orient1")
    ubcalc.set_u(I)

    constraints = Constraints()
    constraints.nu = 0
    constraints.psi = 90
    constraints.phi = 90

    hklcalc = HklCalculation(ubcalc, constraints)
    assert (
        str(hklcalc)
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


def test_get_position_with_radians(cubic: HklCalculation):
    cubic.constraints = Constraints({"delta": 60, "a_eq_b": True, "mu": 0})

    case = Case("100", (1, 0, 0), (0, pi / 3, 0, pi / 6, 0, 0))

    convert_position_to_hkl_and_hkl_to_position(cubic, case, asdegrees=False)


@pytest.mark.parametrize(
    ("constraints"),
    [{}, {"mu": 1, "eta": 1, "bisect": True}, {"bisect": True, "eta": 34, "naz": 3}],
)
def test_get_position_raises_exception_if_badly_constrained(
    cubic: HklCalculation, constraints: Dict[str, Union[float, bool]]
):
    cubic.constraints = Constraints(constraints)

    with pytest.raises(DiffcalcException):
        cubic.get_position(1, 0, 0, 1)


@pytest.mark.parametrize(
    ("constraints"),
    ({"naz": 3, "alpha": 1, "phi": 45}, {"eta": 20, "phi": 34, "delta": 10000}),
)
def test_get_position_raises_exception_if_no_solutions_found(
    constraints: Dict[str, Union[float, bool]]
):
    ubcalc = UBCalculation()
    ubcalc.set_lattice("cube", 1)
    ubcalc.add_reflection((0, 0, 1), Position(1, 2, 3, 4, 5, 6), 12)
    ubcalc.add_orientation((0, 1, 0), (0, 1, 0), Position(1, 2, 3, 4, 5, 6), "orient")
    ubcalc.calc_ub(0, "orient")

    hklcalc = HklCalculation(ubcalc, Constraints(constraints))

    with pytest.raises(DiffcalcException):
        hklcalc.get_position(1, 0, 0, 1)


@pytest.mark.parametrize(
    ("expected_position", "constraints"),
    [
        ((0, 0, 90, 0, 0, 180), {"mu": 0, "eta": 0, "delta": 0}),
        ((45, 0, 90, 0, -45, 90), {"omega": 0, "bisect": True, "delta": 0}),
        ((0, 0, 90, 0, 0, 180), {"mu": 0, "bisect": True, "delta": 0}),
        ((45, 0, 90, 0, -45, 90), {"eta": 0, "bisect": True, "delta": 0}),
        ((0, 0, 90, -180, 0, 0), {"chi": 0, "phi": 0, "delta": 0}),
        ((0, 0, 90, 180, 0, 0), {"mu": 0, "phi": 0, "delta": 0}),
        ((0, 90, 0.0, 90.0, 90.0, 0.0), {"mu": 0, "chi": 90.0, "phi": 0.0}),
        ((180, 0, 90, 0, 180, 0), {"eta": 0, "phi": 0, "delta": 0}),
        ((0, 0, 90, 0, 0, 180), {"eta": 0, "chi": 0, "delta": 0}),
        ((0, 0, 90, 0, 0, 180), {"mu": 0, "eta": 0, "delta": 0}),
        ((0, 90, 0, 0, 90, 180), {"mu": 0, "eta": 0, "nu": 0}),
        ((0, 0, 90, 0, 0, 180), {"mu": 0, "eta": 0, "qaz": 0}),
    ],
)
def test_get_position_one_det_two_samp(
    cubic: HklCalculation,
    expected_position: Tuple[float, float, float, float, float, float],
    constraints: Dict[str, Union[float, bool]],
):
    cubic.constraints = Constraints(constraints)

    all_positions = cubic.get_position(0, 1, 1, 1)

    assert all_positions[0][0].astuple == pytest.approx(expected_position)


@pytest.mark.parametrize(
    ("expected_position", "constraints"),
    [
        ((0, 0, 90, 0, 0, -180), {"psi": 0, "mu": 0, "delta": 0}),
        ((0, 90, 0, 0, 90, -180), {"psi": 0, "mu": 0, "nu": 0}),
        ((0, 0, 90, 0, 0, -180), {"psi": 0, "mu": 0, "qaz": 0}),
        ((0, 0, 90, 0, 0, -180), {"psi": 0, "mu": 0, "naz": 0}),
        ((180, 0, 90, 0, 180, 0), {"psi": 0, "phi": 0, "delta": 0}),
        ((0, 0, 90, 0, 0, -180), {"psi": 0, "eta": 0, "delta": 0}),
        ((-10, 0, 90, 90, 10, 90), {"psi": 0, "chi": 10, "delta": 0}),
        ((0, 0, 90, 90, 90, -90), {"betaout": 0, "mu": 0, "delta": 0}),
    ],
)
def test_get_position_one_det_one_ref_one_samp(
    cubic: HklCalculation,
    expected_position: Tuple[float, float, float, float, float, float],
    constraints: Dict[str, Union[float, bool]],
):
    cubic.constraints = Constraints(constraints)

    all_positions = cubic.get_position(0, 1, 1, 1)

    assert all_positions[0][0].astuple == pytest.approx(expected_position)


@pytest.mark.parametrize(
    ("expected_position", "constraints"),
    [
        ((0, -90, 0, -180, 90, 0), {"chi": 90, "phi": 0, "psi": 0}),
        ((90, 90, 0, 0, 90, -90), {"mu": 90, "eta": 0, "psi": 0}),
        ((0, 0, 90, 0, 0, -180), {"chi": 0, "eta": 0, "psi": 0}),
        ((0, 90, 0, 0, 90, -180), {"chi": 90, "mu": 0, "psi": 0}),
        ((0, 0, 90, 90, 0, 90), {"mu": 0, "phi": 90, "psi": 0}),
        ((90, -90, 0, 0, -90, 90), {"eta": 0, "phi": 90, "psi": 0}),
        ((-45, -90, 0, -180, 90, 45), {"chi": 90, "phi": 45, "betain": 0}),
    ],
)
def test_get_position_one_ref_two_samp(
    cubic: HklCalculation,
    expected_position: Tuple[float, float, float, float, float, float],
    constraints: Dict[str, Union[float, bool]],
):
    cubic.constraints = Constraints(constraints)

    all_positions = cubic.get_position(0, 1, 1, 1)

    assert all_positions[0][0].astuple == pytest.approx(expected_position)


@pytest.mark.parametrize(
    ("expected_position", "constraints"),
    [
        ((90, 90, 0, 90, 0, 0), {"eta": 90, "chi": 0, "phi": 0}),
        ((45, 45, -90, 90, 45, 45), {"mu": 45, "chi": 45, "phi": 45}),
        ((45, 0, 90, 90, -45, 90), {"mu": 45, "eta": 90, "phi": 90}),
        ((90, 90, 0, 0, 90, 270), {"mu": 90, "eta": 0, "chi": 90}),
    ],
)
def test_get_position_three_samp(
    cubic: HklCalculation,
    expected_position: Tuple[float, float, float, float, float, float],
    constraints: Dict[str, Union[float, bool]],
):
    cubic.constraints = Constraints(constraints)

    all_positions = cubic.get_position(0, 1, 1, 1)

    assert all_positions[0][0].astuple == pytest.approx(expected_position)


def test_scattering_angle_parallel_to_xray_beam_yields_nan_psi_virtual_angle(
    cubic_ub: UBCalculation,
):
    configure_ub(cubic_ub)
    cubic_ub.n_hkl = (1, -1, 0)  # type: ignore
    hklcalc = HklCalculation(cubic_ub, Constraints({"nu": 0, "chi": 0, "phi": 0}))

    virtual_angles = hklcalc.get_virtual_angles(Position(0, 180, 0, 90, 0, 0))
    assert isnan(virtual_angles["psi"])


@pytest.mark.parametrize(
    ("constraints"),
    [
        {"delta": 60, "a_eq_b": True, "mu": 0},
        {"psi": 90, "mu": 0, "nu": 0},
        {"bin_eq_bout": True, "mu": 0, "qaz": 90},
    ],
)
def test_fails_for_parallel_vectors(
    cubic: HklCalculation, constraints: Dict[str, Union[float, bool]]
):
    cubic.constraints.asdict = constraints

    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("constraints", "zrot", "yrot", "case", "n_hkl"),
    [
        (
            {"delta": 90, "beta": 0, "phi": 0},
            0,
            0,
            Case("sqrt(2)00", (sqrt(2), 0, 0), (0, 90, 0, 45, 0, 0)),
            (1, -1, 0),
        ),
        (
            {"nu": 60, "psi": 0, "phi": 0},
            -1,
            2,
            Case("100", (1, 0, 0), (118, 0, 60, 91, 180, 0)),
            None,
        ),
    ],
)
def test_redundant_solutions_when_calculating_remaining_detector_angles(
    cubic_ub: UBCalculation,
    constraints: Dict[str, float],
    zrot: float,
    yrot: float,
    case: Case,
    n_hkl: Optional[Tuple[float, float, float]],
):
    configure_ub(cubic_ub, zrot, yrot)
    hklcalc = HklCalculation(cubic_ub, Constraints(constraints))
    if n_hkl:
        hklcalc.ubcalc.n_hkl = (1, -1, 0)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
    hkl_mocked_constraints: HklCalculation,
    angle: str,
    expected_values_for_positions: Dict[
        float, List[Tuple[float, float, float, float, float, float]]
    ],
):
    for expected_value, positions in expected_values_for_positions.items():
        for each_position in positions:
            virtual_angles = hkl_mocked_constraints.get_virtual_angles(
                Position(*each_position)
            )
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
    hkl_mocked_constraints: HklCalculation,
    angle: str,
    positions: List[Tuple[float, float, float, float, float, float]],
):
    for each_position in positions:
        calculated_value = hkl_mocked_constraints.get_virtual_angles(
            Position(*each_position)
        )[angle]
        assert isnan(calculated_value)
