import itertools
from math import cos, degrees, pi, radians, sin, sqrt
from typing import Dict, Tuple, Union

import numpy as np
import pytest
from diffcalc import Q
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
def hklcalc() -> HklCalculation:
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore
    return HklCalculation(ubcalc, Constraints())


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


@pytest.mark.parametrize(
    ("expected_position", "constraints"),
    [
        ((0, 0, 90, 0, 0, -180), {"psi": 0, "mu": 0, "delta": 0}),
        ((0, 90, 0, 0, 90, -180), {"psi": 0, "mu": 0, "nu": 0}),
        ((0, 0, 90, 0, 0, -180), {"psi": 0, "mu": 0, "qaz": 0}),
        ((0, 0, 90, 0, 0, -180), {"psi": 0, "mu": 0, "naz": 0}),
        ((180, 0, 90, 0, 180, 0), {"psi": 0, "phi": 0, "delta": 0}),
        ((0, 0, 90, 0, 0, -180), {"psi": 0, "eta": 0, "delta": 0}),
        ((-10, 0, 90, 90, 10, 90), {"psi": 0, "chi": Q(10, "deg"), "delta": 0}),
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

    assert tuple(
        degrees(item) for item in all_positions[0][0].astuple
    ) == pytest.approx(expected_position)


@pytest.mark.parametrize(
    ("case"),
    (
        Case("100", (1, 0, 0), (-90, 60, 0, 0, 30, 0)),
        Case("010", (0, 1, 0), (90, 60, 0, 0, 150, -90)),
        Case("110", (1, 1, 0), (-90, 90, 0, 0, 45, 45)),
        Case("111", (1, 1, 1), (90, 120, 0, 0, 84.7356, -135)),
    ),
)
def test_fixed_naz_psi_eta(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"naz": Q(90, "deg"), "psi": 0, "eta": 0})

    convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


def test_fixed_naz_psi_eta_fails_for_parallel_vectors(cubic: HklCalculation):
    cubic.constraints = Constraints({"naz": Q(90, "deg"), "psi": 0, "eta": 0})
    case = Case("001", (0, 0, 1), (30, 0, 60, 90, 0, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


@pytest.mark.parametrize(
    ("case", "constraints"),
    itertools.product(
        [
            Case("100", (1, 0, 0), (-90, 60, 0, 90, 90, -60)),
            Case(
                "100-->001",
                (cos(radians(4)), sin(radians(4)), 0),
                (-90, 60, 0, 90, 90, -56),
            ),
            Case("010", (0, 1, 0), (90, 60, 0, -90, 90, -150)),
            Case(
                "001-->100",
                (cos(radians(86)), sin(radians(86)), 0),
                (90, 60, 0, 90, 90, -34),
            ),
        ],
        [
            {"a_eq_b": True, "qaz": Q(90, "deg"), "chi": Q(90, "deg")},
            {"a_eq_b": True, "nu": 0, "chi": Q(90, "deg")},
        ],
    ),
)
def test_det_ref_and_chi_constrained(
    cubic: HklCalculation, case: Case, constraints: Dict[str, Union[float, bool]]
):
    cubic.constraints = Constraints(constraints)

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case", "constraints"),
    itertools.product(
        [
            Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0)),
        ],
        [
            {"a_eq_b": True, "qaz": Q(90, "deg"), "chi": Q(90, "deg")},
            {"a_eq_b": True, "nu": 0, "chi": Q(90, "deg")},
        ],
    ),
)
def test_det_ref_and_chi_constrained_fails_for_parallel_vectors(
    cubic: HklCalculation, case: Case, constraints: Dict[str, Union[float, bool]]
):
    cubic.constraints = Constraints(constraints)

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case", "zrot", "yrot", "constraints"),
    itertools.product(
        [
            Case("100", (1, 0, 0), (120, 0, 60, 90, 180, 0)),
            Case(
                "100-->001",
                (cos(radians(4)), 0, sin(radians(4))),
                (124, 0, 60, 90, 180, 0),
            ),
            Case("010", (0, 1, 0), (120, 0, 60, 0, 180, 0)),
            Case("001", (0, 0, 1), (30, 0, 60, 90, 0, 0)),
            Case(
                "001-->100",
                (cos(radians(86)), 0, sin(radians(86))),
                (26, 0, 60, 90, 0, 0),
            ),
        ],
        [
            -1,
            1,
        ],
        [2],
        [
            {"delta": 0, "psi": 0, "phi": 0},
            {"nu": Q(60, "deg"), "psi": 0, "phi": 0},
        ],
    ),
)
def test_ref_and_fixed_delta_phi_0(
    cubic_ub: UBCalculation,
    case: Case,
    zrot: float,
    yrot: float,
    constraints: Dict[str, float],
):
    configure_ub(cubic_ub, zrot, yrot)
    hklcalc = HklCalculation(cubic_ub, Constraints(constraints))

    if case.name.startswith("100"):
        new_position = (
            case.position[0] - yrot,
            *case.position[1:3],
            case.position[3] - zrot,
            *case.position[4:],
        )
    elif case.name.startswith("001"):
        new_position = (
            case.position[0] - yrot,
            *case.position[1:3],
            case.position[3] + zrot,
            *case.position[4:],
        )
    else:
        new_position = (*case.position[:3], case.position[3] - zrot, *case.position[4:])

    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


@pytest.mark.parametrize(
    ("constraints"),
    [
        {"qaz": pi / 2, "alpha": pi / 2, "phi": 0},
        {"delta": pi / 2, "beta": 0, "phi": 0},
        {"delta": pi / 2, "betain": 0, "phi": 0},
    ],
)
def test_alpha_90(cubic: HklCalculation, constraints: Dict[str, float]):
    cubic.constraints = Constraints(constraints)
    cubic.ubcalc.n_hkl = (1, -1, 0)

    case = Case("sqrt(2)00", (sqrt(2), 0, 0), (0, 90, 0, 45, 0, 0))

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("zrot, constraints"),
    itertools.product(
        [1, -1],
        [
            {"a_eq_b": True, "mu": 0, "nu": 0},
            {"psi": pi / 2, "mu": 0, "nu": 0},
            {"a_eq_b": True, "mu": 0, "qaz": pi / 2},
        ],
    ),
)
def test_small_zrot(
    cubic_ub: UBCalculation,
    zrot: float,
    constraints: Dict[str, float],
):
    configure_ub(cubic_ub, zrot, 0)
    hklcalc = HklCalculation(cubic_ub, Constraints(constraints))

    pos = (0, 60, 0, 30, 0, 90 + zrot)
    case_with_rotation = Case("", (0, 1, 0), pos)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


@pytest.mark.parametrize(
    ("case, zrot, constraints"),
    itertools.product(
        [
            Case("100", (1, 0, 0), (0, 60, 0, 30, 0, 0)),
            Case(
                "100-->001",
                (cos(radians(4)), 0, sin(radians(4))),
                (0, 60, 0, 30, 4, 0),
            ),
            Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0)),
            Case("010", (0, 1, 0), (0, 60, 0, 30, 0, 90)),
            Case(
                "0.1 0 1.5",
                (0.1, 0, 1.5),
                (
                    0,
                    97.46959231642,
                    0,
                    97.46959231642 / 2,
                    86.18592516571,
                    0,
                ),
            ),
            Case(
                "001->100",
                (cos(radians(86)), 0, sin(radians(86))),
                (0, 60, 0, 30, 86, 0),
            ),
        ],
        [1, -1],
        [
            {"a_eq_b": True, "mu": 0, "nu": 0},
            {"psi": pi / 2, "mu": 0, "nu": 0},
            {"a_eq_b": True, "mu": 0, "qaz": pi / 2},
        ],
    ),
)
def test_various_constraints_and_small_zrot_yrot(
    cubic_ub: UBCalculation,
    case: Case,
    zrot: float,
    constraints: Dict[str, float],
):
    configure_ub(cubic_ub, zrot, 2)
    hklcalc = HklCalculation(cubic_ub, Constraints(constraints))

    pos = case.position
    if case.name != "010":
        new_position = (*pos[:4], pos[4] - 2, pos[5] + zrot)
    else:
        new_position = (*pos[:5], pos[5] + zrot)

    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


@pytest.mark.parametrize(
    ("case", "zrot", "constraints"),
    itertools.product(
        [
            Case("100", (1, 0, 0), (0, 60, 0, 30, 0, 0)),
            Case(
                "100-->001",
                (cos(radians(4)), 0, sin(radians(4))),
                (0, 60, 0, 30, 4, 0),
            ),
            Case("010", (0, 1, 0), (0, 60, 0, 30, 0, 90)),
            Case(
                "001->100",
                (cos(radians(86)), 0, sin(radians(86))),
                (0, 60, 0, 30, 86, 0),
            ),
        ],
        [0, 2, -2, 45, -45, 90, -90],
        [
            {"delta": pi / 3, "a_eq_b": True, "mu": 0},
            {"a_eq_b": True, "mu": 0, "nu": 0},
            {"psi": pi / 2, "mu": 0, "nu": 0},
            {"a_eq_b": True, "mu": 0, "qaz": pi / 2},
        ],
    ),
)
def test_various_constraints_and_zrots(
    cubic_ub: UBCalculation, case: Case, zrot: float, constraints: Dict[str, float]
):
    configure_ub(cubic_ub, zrot, 0)
    hklcalc = HklCalculation(cubic_ub, Constraints(constraints))

    pos = case.position
    new_position = (*pos[:5], pos[5] + zrot)
    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


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
    cubic: HklCalculation,
    case: Case,
    constraints: Dict[str, Union[float, bool]],
):
    cubic.constraints = Constraints(constraints)

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


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
    cubic_ub: UBCalculation,
    case: Case,
    yrot: float,
    constraints: Dict[str, Union[float, bool]],
):
    configure_ub(cubic_ub, 1, yrot)
    hklcalc = HklCalculation(cubic_ub, Constraints(constraints))

    if case.name.startswith("100"):
        case_with_rotation = Case(
            case.name,
            case.hkl,
            (*case.position[:4], case.position[4] + yrot, case.position[5] + 1),
        )
    else:
        case_with_rotation = Case(
            case.name,
            case.hkl,
            (*case.position[:4], case.position[4] - yrot, case.position[5] + 1),
        )

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
    cubic_ub: UBCalculation, yrot: float, constraints: Dict[str, Union[bool, float]]
):
    configure_ub(cubic_ub, 1, yrot)
    hklcalc = HklCalculation(cubic_ub, Constraints(constraints))

    case = Case("010", (0, 1, 0), (30, 0, 60, 0, 90, -89))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case"),
    [
        Case(
            "7_9_13",
            (0.7, 0.9, 1.3),
            (0, 27.352179, 0, 13.676090, 37.774500, 53.96550),
            1.24,
        ),
        Case("100", (1, 0, 0), (0, 18.580230, 0, 9.290115, -2.403500, 3.589000), 1.24),
        Case("010", (0, 1, 0), (0, 18.580230, 0, 9.290115, 0.516000, 93.567000), 1.24),
        Case(
            "110", (1, 1, 0), (0, 26.394192, 0, 13.197096, -1.334500, 48.602000), 1.24
        ),
    ],
)
def test_against_spec_sixc_b16_270608(hklcalc: HklCalculation, case: Case):
    hklcalc.ubcalc.set_lattice("name", 3.8401, 5.43072)
    u_matrix = np.array(
        (
            (0.997161, -0.062217, 0.042420),
            (0.062542, 0.998022, -0.006371),
            (-0.041940, 0.009006, 0.999080),
        )
    )
    hklcalc.ubcalc.set_u(u_matrix)
    hklcalc.constraints.asdict = {"a_eq_b": True, "mu": 0, "nu": 0}

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 2)


@pytest.mark.parametrize(
    ("case"),
    [
        Case("", (1, 1, 1), (0.0, 21.8980, 0.0, 11, 89.8419, 10.8224), 1.239842),
        Case("", (0, 0, 2), (0.0, 25.3375, 0.0, 19.2083, 35.4478, 81.2389), 1.239842),
    ],
)
def test_i16_ga_as_example(hklcalc: HklCalculation, case: Case):
    hklcalc.ubcalc.set_lattice("xtal", 5.65315)
    hklcalc.ubcalc.set_u(
        np.array(
            [
                [-0.71021455, 0.70390373, 0.01071626],
                [-0.39940627, -0.41542895, 0.81724747],
                [0.57971538, 0.5761409, 0.57618724],
            ]
        )
    )
    hklcalc.constraints.asdict = {"qaz": pi / 2, "alpha": Q(11, "deg"), "mu": 0}

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 3)


@pytest.mark.parametrize(
    ("case"),
    [
        Case(
            "",
            (0.0, 0.2, 0.25),
            (0, 62.44607, 0, 28.68407, 90.0 - 0.44753, -9.99008),
            12.39842 / 0.650,
        ),
        Case(
            "",
            (0.25, 0.2, 0.1),
            (0, 108.03033, 0, 3.03132, 90 - 7.80099, 87.95201),
            12.39842 / 0.650,
        ),
    ],
)
def test_i21(hklcalc: HklCalculation, case: Case):
    hklcalc.ubcalc.set_lattice("xtal", 3.78, 20.10)
    hklcalc.ubcalc.n_phi = (0, 0, 1)

    hklcalc.ubcalc.set_u(
        np.array(((1.0, 0.0, 0.0), (0.0, 0.18482, -0.98277), (0.0, 0.98277, 0.18482)))
    )

    hklcalc.constraints.asdict = {"psi": Q(10, "deg"), "mu": 0, "nu": 0}

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 3)
