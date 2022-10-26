import itertools
from dataclasses import dataclass
from math import isnan, pi, radians, sqrt
from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import Mock

import numpy as np
import pytest
from diffcalc import Q, ureg
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
def tetragonal_ub() -> UBCalculation:
    ubcalc = UBCalculation()

    ubcalc.set_lattice(name="test", a=4.913, c=5.405)
    ubcalc.add_reflection(
        (0, 0, 1),
        Position(Q(7.31, "deg"), 0, Q(10.62, "deg"), 0, 0, 0),
        12.39842,
        "refl1",
    ),
    ubcalc.add_orientation((0, 1, 0), (0, 1, 0), tag="plane")

    ubcalc.n_hkl = (1.0, 0, 0)  # type: ignore

    return ubcalc


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

    position: Position = (
        Position(*case.position * ureg.deg) if asdegrees else Position(*case.position)
    )
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
    ubcalc.add_reflection(
        (0, 0, 1), Position(0, Q(60, "deg"), 0, Q(30, "deg"), 0, 0), 12.4, "ref1"
    )
    ubcalc.add_orientation(
        (0, 1, 0),
        (0, 1, 0),
        Position(Q(1, "deg"), 0, 0, 0, Q(2, "deg"), 0),
        "orient1",
    )
    ubcalc.set_u(I)

    constraints = Constraints({"nu": 0.0, "psi": Q(90, "deg"), "phi": Q(90, "deg")})

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
    cubic.constraints = Constraints({"delta": 60 * ureg.deg, "a_eq_b": True, "mu": 0.0})

    case = Case("100", (1, 0, 0), (0, pi / 3, 0, pi / 6, 0, 0))

    convert_position_to_hkl_and_hkl_to_position(cubic, case, asdegrees=False)


@pytest.mark.parametrize(
    ("constraints"),
    [
        {},
        {"mu": 1.0 * ureg.deg, "eta": 1 * ureg.deg, "bisect": True},
        {"bisect": True, "eta": 34 * ureg.deg, "naz": 3 * ureg.deg},
    ],
)
def test_get_position_raises_exception_if_badly_constrained(
    cubic: HklCalculation, constraints: Dict[str, Union[float, bool]]
):
    cubic.constraints = Constraints(constraints)

    with pytest.raises(DiffcalcException):
        cubic.get_position(1, 0, 0, 1)


@pytest.mark.parametrize(
    ("constraints"),
    (
        {"naz": 3 * ureg.deg, "alpha": 1 * ureg.deg, "phi": 45 * ureg.deg},
        {"eta": 20 * ureg.deg, "phi": 34 * ureg.deg, "delta": 10000 * ureg.deg},
    ),
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
    ("zrot", "constraints"),
    itertools.product(
        [0, 2, -2, 45, -45, 90, -90],
        [
            {"delta": 60 * ureg.deg, "a_eq_b": True, "mu": 0.0},
            {"a_eq_b": True, "mu": 0.0, "nu": 0.0},
            {"psi": 90 * ureg.deg, "mu": 0.0, "nu": 0.0},
            {"a_eq_b": True, "mu": 0.0, "qaz": 90 * ureg.deg},
        ],
    ),
)
def test_fails_for_parallel_vectors(
    cubic_ub: UBCalculation, zrot: float, constraints: Dict[str, Union[float, bool]]
):
    """Confirm that a hkl of (0,0,1) fails for a_eq_b=True.
    By default, the reference vector is (0,0,1). A parallel vector to this should
    cause failure
    """
    configure_ub(cubic_ub, zrot, 0)
    hklcalc = HklCalculation(cubic_ub, Constraints(constraints))

    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0 + zrot))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


def test_scattering_angle_parallel_to_xray_beam_yields_nan_psi_virtual_angle(
    cubic_ub: UBCalculation,
):
    configure_ub(cubic_ub)
    cubic_ub.n_hkl = (1, -1, 0)  # type: ignore
    hklcalc = HklCalculation(cubic_ub, Constraints({"nu": 0.0, "chi": 0.0, "phi": 0.0}))
    virtual_angles = hklcalc.get_virtual_angles(
        Position(0, 180 * ureg.deg, 0, 90 * ureg.deg, 0, 0)
    )
    assert isnan(virtual_angles["psi"])


@pytest.mark.parametrize(
    ("constraints", "zrot", "yrot", "case", "n_hkl"),
    [
        (
            {"delta": 90 * ureg.deg, "beta": 0.0, "phi": 0.0},
            0,
            0,
            Case("sqrt(2)00", (sqrt(2), 0, 0), (0, 90, 0, 45, 0, 0)),
            (1, -1, 0),
        ),
        (
            {"nu": 60 * ureg.deg, "psi": 0.0, "phi": 0.0},
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
    cubic: HklCalculation,
    case: Case,
    constraints: Dict[str, Union[float, bool]],
):
    cubic.constraints = Constraints(constraints)

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case"),
    [
        Case("", (1, 0, 0), (0, 60, 0, 30, 1, 1)),
        Case("", (1.00379806, -0.006578435, 0.08682408), (0, 60, 10, 30, 1, 1)),
        Case("", (0.99620193, 0.0065784359, 0.08682408), (10, 60, 0, 30, 1, 1)),
        Case(
            "", (1.01174189, 0.02368622, 0.06627361), (0.9, 60.9, 2.9, 30.9, 2.9, 1.9)
        ),
    ],
)
def test_get_hkl(cubic_ub: UBCalculation, case: Case):
    u_matrix = np.array(
        (
            (0.9996954135095477, -0.01745240643728364, -0.017449748351250637),
            (0.01744974835125045, 0.9998476951563913, -0.0003045864904520898),
            (0.017452406437283505, -1.1135499981271473e-16, 0.9998476951563912),
        )
    )
    cubic_ub.set_u(u_matrix)

    hkl = HklCalculation(cubic_ub, Constraints()).get_hkl(
        Position(*case.position * ureg.deg), 1.0
    )

    assert np.all(np.round(hkl, 7) == np.round(case.hkl, 7))


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
                Position(*each_position * ureg.deg)
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
            Position(*each_position * ureg.deg)
        )[angle]
        assert isnan(calculated_value)
