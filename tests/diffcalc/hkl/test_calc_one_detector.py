import itertools
from math import cos, pi, radians, sin
from typing import Dict, Tuple, Union

import numpy as np
import pytest
from diffcalc import ureg
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.ub.crystal import LatticeParams
from diffcalc.util import DiffcalcException, I

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

    ubcalc.set_lattice("Cubic", LatticeParams(1))
    configure_ub(ubcalc)

    return HklCalculation(ubcalc, Constraints())


@pytest.fixture
def cubic_ub() -> UBCalculation:
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore

    ubcalc.set_lattice("Cubic", LatticeParams(1))
    return ubcalc


@pytest.mark.parametrize(
    ("expected_position", "constraints"),
    [
        ((0, 0, 90, 0, 0, 180), {"mu": 0, "eta": 0, "delta": 0}),
        ((45, 0, 90, 0, -45, 90), {"omega": 0, "bisect": True, "delta": 0}),
        ((0, 0, 90, 0, 0, 180), {"mu": 0, "bisect": True, "delta": 0}),
        ((45, 0, 90, 0, -45, 90), {"eta": 0, "bisect": True, "delta": 0}),
        ((0, 0, 90, -180, 0, 0), {"chi": 0, "phi": 0, "delta": 0}),
        ((0, 0, 90, 180, 0, 0), {"mu": 0, "phi": 0, "delta": 0}),
        ((0, 90, 0.0, 90.0, 90.0, 0.0), {"mu": 0, "chi": pi / 2, "phi": 0.0}),
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

    calculated_pos = all_positions[0][0]

    assert np.all(
        [
            radians(expected_position[i])
            == pytest.approx(float(calculated_pos.astuple[i]))
            for i in range(6)
        ]
    )


@pytest.mark.parametrize(
    ("case", "yrot"),
    itertools.product(
        [
            Case("100", (1, 0, 0), (30, 0, 60, 0, -90, 0)),
            Case(
                "100-->001",
                (cos(radians(4)), 0, sin(radians(4))),
                (30, 0, 60, 0, -90 + 4, 0),
            ),
            Case("001", (0, 0, 1), (30, 0, 60, 0, 0, 0)),
            Case(
                "001-->100",
                (cos(radians(86)), 0, sin(radians(86))),
                (30, 0, 60, 0, -4, 0),
            ),
        ],
        [0, 2, -2, 45, -45, 90, -90],
    ),
)
def test_fixed_delta_eta_phi_0(cubic_ub: UBCalculation, case: Case, yrot: float):
    configure_ub(cubic_ub, 0, yrot)
    hklcalc = HklCalculation(cubic_ub, Constraints({"eta": 0, "delta": 0, "phi": 0}))

    new_position = (*case.position[:4], case.position[4] - yrot, case.position[5])
    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


def test_fixed_delta_eta_phi_0_fails_as_non_unique_chi(cubic: HklCalculation):
    case = Case("010", (0, 1, 0), (120, 0, 60, 0, 0, 0))

    cubic.constraints = Constraints({"eta": 0, "delta": 0, "phi": 0})

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case"),
    (
        Case("100", (1, 0, 0), (0, 0, 60, 0, -90, 30)),
        Case("010", (0, 1, 0), (90, 0, 60, 0, -90, 30)),
        Case("001", (0, 0, 1), (30, 0, 60, 0, 0, 30)),
    ),
)
def test_fixed_delta_eta_phi_30(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"eta": 0, "delta": 0, "phi": pi / 6})

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case", "zrot"),
    itertools.product(
        [
            Case("100", (1, 0, 0), (120, 0, 60, 0, 0, -90)),
            Case(
                "100-->001",
                (cos(radians(4)), 0, sin(radians(4))),
                (120 - 4, 0, 60, 0, 0, -90),
            ),
            Case(
                "010",
                (0, 1, 0),
                (120, 0, 60, 0, 0, 0),
            ),
            Case(
                "001-->100",
                (cos(radians(86)), 0, sin(radians(86))),
                (30 - 4, 0, 60, 0, 0, 90),
            ),
        ],
        [0, 2, -2],
    ),
)
def test_fixed_delta_eta_chi_0(cubic_ub: UBCalculation, case: Case, zrot: float):
    configure_ub(cubic_ub, zrot, 0)
    hklcalc = HklCalculation(cubic_ub, Constraints({"eta": 0, "delta": 0, "chi": 0}))

    new_position = (*case.position[:5], case.position[5] + zrot)
    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


# phi || q
def test_fixed_delta_eta_chi_0_fails_for_degenerate_case(cubic: HklCalculation):
    cubic.constraints = Constraints({"eta": 0, "delta": 0, "chi": 0})

    case = Case("001", (0, 0, 1), (30, 0, 60, 0, 0, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case"),
    [
        Case("100", (1, 0, 0), (120, 0, 60, 0, 30, -90)),
        Case(
            "100-->001",
            (-sin(radians(30)), 0, cos(radians(30))),
            (30, 0, 60, 0, 30, 0),
        ),
        Case("010", (0, 1, 0), (120, 0, 60, 0, 30, 0)),
    ],
)
def test_fixed_delta_eta_chi_30(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"eta": 0, "delta": 0, "chi": pi / 6})

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case"),
    [
        Case("100", (1, 0, 0), (0, 60, 0, 120, 30, -90)),
        Case("010", (0, 1, 0), (0, 60, 0, 120, 30, 0)),
        Case(
            "100-->010",
            (sin(radians(30)), cos(radians(30)), 0),
            (0, 60, 0, 120, 30, -30),
        ),
    ],
)
def test_fixed_gamma_mu_chi_30(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"mu": 0, "nu": 0, "chi": pi / 6})

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case", "zrot"),
    itertools.product(
        [
            Case("100", (1, 0, 0), (0, 60, 0, 120, 90, -90)),
            Case("010", (0, 1, 0), (0, 60, 0, 120, 90, 0)),
            Case(
                "010-->100",
                (sin(radians(4)), cos(radians(4)), 0),
                (0, 60, 0, 120, 90, -4),
            ),
            Case(
                "100-->010",
                (sin(radians(86)), cos(radians(86)), 0),
                (0, 60, 0, 120, 90, -90 + 4),
            ),
        ],
        [0, 2, -2],
    ),
)
def test_fixed_gamma_mu_chi_90(cubic_ub: UBCalculation, case: Case, zrot: float):
    configure_ub(cubic_ub, zrot, 0)
    hklcalc = HklCalculation(cubic_ub, Constraints({"mu": 0, "nu": 0, "chi": pi / 2}))

    case_with_rotation = Case(
        case.name, case.hkl, (*case.position[:5], case.position[5] + zrot)
    )

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


def test_fixed_gamma_mu_chi_90_fails_for_non_unique_phi(cubic: HklCalculation):
    cubic.constraints = Constraints({"mu": 0, "nu": 0, "chi": pi / 2})

    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case"),
    (
        Case("101", (1, 0, 1), (0, 90, 0, 45, 45, 0)),
        Case("10m1", (1, 0, -1), (0, 90, 0, 45, -45, 0)),
        Case("011", (0, 1, 1), (0, 90, 0, 45, 45, 90)),
        Case(
            "100-->001",
            (sin(radians(4)), 0, cos(radians(4))),
            (0, 60, 0, 30, 90 - 4, 0),
        ),
        Case("010", (0, 1, 0), (0, 60, 0, 30, 0, 90)),
        Case(
            "0.1 0 1.5",
            (0.1, 0, 1.5),
            (0, 97.46959231642, 0, 48.73480, 86.18593, 0),
        ),
        Case(
            "010-->001",
            (0, cos(radians(86)), sin(radians(86))),
            (0, 60, 0, 30, 90 - 4, 90),
        ),
    ),
)
def test_with_bisect_and_nu_0_omega_0(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"bisect": True, "nu": 0, "omega": 0})

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


def test_with_bisect_and_nu_0_omega_0_fails_for_non_unique_phi(cubic: HklCalculation):
    cubic.constraints = Constraints({"bisect": True, "nu": 0, "omega": 0})
    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case"),
    [
        Case("011", (0, 1, 1), (90, 90, 0, 0, 0, 90)),
        Case("100-->001", (sin(radians(4)), 0, cos(radians(4))), (90, 60, 0, 0, 56, 0)),
        Case("010", (0, 1, 0), (90, 60, 0, 0, -30, 90)),
        Case(
            "0.1, 0, 1.5", (0.1, 0, 1.5), (90, 97.46959231642, 0, 0, 37.45112900750, 0)
        ),
        Case(
            "010-->001", (0, cos(radians(86)), sin(radians(86))), (90, 60, 0, 0, 56, 90)
        ),
    ],
)
def test_fixed_mu_eta(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"nu": 0, "mu": pi / 2, "eta": 0})

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


def test_fixed_mu_eta_fails_for_non_unique_phi(cubic: HklCalculation):
    cubic.constraints = Constraints({"nu": 0, "mu": pi / 2, "eta": 0})
    case = Case("001", (0, 0, 1), (90, 60, 0, 0, 60, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


def test_ttheta_180(cubic: HklCalculation):
    cubic.constraints = Constraints({"nu": 0, "chi": 0, "phi": 0})
    cubic.n_hkl = (1, -1, 0)  # type: ignore

    case = Case("200", (2, 0, 0), (0, 180, 0, 90, 0, 0))

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case"),
    [
        Case("110", (1, 1, 0), (-90, 90, 0, 90, 90, 0)),
        Case(
            "100-->001",
            (sin(radians(4)), 0, cos(radians(4))),
            (
                -8.01966360660,
                60,
                0,
                29.75677306273,
                90,
                0,
            ),
        ),
        Case("010", (0, 1, 0), (0, 60, 0, 120, 90, 0)),
        Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0)),
        Case(
            "0.1,0,1.5",
            (0.1, 0, 1.5),
            (
                -5.077064540005,
                97.46959231642,
                0,
                48.62310452627,
                90,
                0,
            ),
        ),
        Case(
            "010-->001",
            (0, cos(radians(86)), sin(radians(86))),
            (0, 60, 0, 34, 90, 0),
        ),
    ],
)
def test_with_chi_phi_constrained(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"nu": 0, "chi": pi / 2, "phi": 0.0})

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case", "yrot"),
    itertools.product(
        [
            Case(
                "100",
                (1, 0, 0),
                (0, 60, 0, 30, 0, 0),
            ),
            Case(
                "100-->001",
                (cos(radians(4)), 0, sin(radians(4))),
                (0, 60, 0, 30, 4, 0),
            ),
            Case(
                "001",
                (0, 0, 1),
                (0, 60, 0, 30, 90, 0),
            ),
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
                "001-->100",
                (cos(radians(86)), 0, sin(radians(86))),
                (0, 60, 0, 30, 86, 0),
            ),
        ],
        [0, 2, -2, 45, -45, 90, -90],
    ),
)
def test_fixed_phi_0(cubic_ub: UBCalculation, case: Case, yrot: float):
    configure_ub(cubic_ub, 0, yrot)
    hklcalc = HklCalculation(cubic_ub, Constraints({"mu": 0, "nu": 0, "phi": 0}))

    pos = case.position
    new_position = (*pos[:3], pos[3], pos[4] - yrot, pos[5])

    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


@pytest.mark.parametrize(
    ("case"),
    (
        Case(
            "100",
            (1, 0, 0),
            (0, 60, 0, 0, 0, 30),
        ),
        Case(
            "001",
            (0, 0, 1),
            (0, 60, 0, 30, 90, 30),
        ),
        Case(
            "0.1 0 1.5",
            (0.1, 0, 1.5),
            (
                0,
                97.46959231642,
                0,
                46.828815370173,
                86.69569481984,
                30,
            ),
        ),
    ),
)
def test_fixed_phi_30(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"mu": 0, "nu": 0, "phi": pi / 6})

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("yrot"),
    [0, 2, -2, 45, -45, 90, -90],
)
def test_fixed_phi_90(cubic_ub: UBCalculation, yrot: float):
    configure_ub(cubic_ub, 0, yrot)
    hklcalc = HklCalculation(cubic_ub, Constraints({"mu": 0, "nu": 0, "phi": pi / 2}))

    case = Case("010", (0, 1, 0), (0, 60, 0, 30, 0, 90))

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
def test_bisect(cubic: HklCalculation, case: Case):
    cubic.constraints = Constraints({"delta": 0, "bisect": True, "omega": 0})

    convert_position_to_hkl_and_hkl_to_position(cubic, case)


def test_bisect_fails_for_non_unique_phi(cubic: HklCalculation):
    cubic.constraints = Constraints({"delta": 0, "bisect": True, "omega": 0})
    case = Case("001", (0, 0, 1), (30, 0, 60, 0, 0, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case)


@pytest.mark.parametrize(
    ("case"),
    [
        Case(
            "001",
            (0, 0, 1),
            (0, 33.07329403295449, 0, 16.536647016477247, 90, -90),
            12.39842 / 1.650,
        ),
        Case(
            "100",
            (1, 0, 0),
            (0, 89.42926563609406, 0, 134.71463281804702, 0, -90),
            12.39842 / 1.650,
        ),  # this one shouldn't pass!!!! TODO: investigate...
        Case(
            "101",
            (1, 0, 1),
            (0, 98.74666191021282, 0, 117.347760720783, 90, -90),
            12.39842 / 1.650,
        ),
    ],
)
def test_three_two_circle_i06_i10(hklcalc: HklCalculation, case: Case):
    hklcalc.ubcalc.set_lattice("xtal", LatticeParams(5.34, 13.2))
    hklcalc.ubcalc.set_u(I)
    hklcalc.constraints.asdict = {"phi": -pi / 2, "nu": 0, "mu": 0}

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case", "constraints"),
    [
        (
            Case(
                "001",
                (0, 0, 1),
                (16.536647016477247, 0, 33.07329403295449, 0, 0, -90),
                12.39842 / 1.650,
            ),
            {"phi": -pi / 2, "delta": 0, "eta": 0},
        ),
        (
            Case(
                "101",
                (1, 0, 1),
                (117.347760720783, 0, 98.74666191021282, 0, 0, -90),
                12.39842 / 1.650,
            ),
            {"phi": -pi / 2, "delta": 0, "eta": 0},
        ),
        (
            Case(
                "100",
                (1, 0, 0),
                (134.71463281804702, 0, 89.42926563609406, 0, 0, -90),
                12.39842 / 1.650,
            ),
            {"phi": -pi / 2, "chi": 0, "delta": 0},
        ),
    ],
)
def test_three_two_circle_i06_i10_horizontal(
    hklcalc: HklCalculation, case: Case, constraints: Dict[str, float]
):
    hklcalc.ubcalc.set_lattice("xtal", LatticeParams(5.34, 13.2))
    hklcalc.ubcalc.set_u(I)
    hklcalc.constraints.asdict = constraints

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case", "constraints"),
    [
        (
            Case(
                "100",
                (1, 0, 0),
                (134.71463281804702, 0, 89.42926563609406, 0, 0, -90),
                12.39842 / 1.650,
            ),
            {"phi": -pi / 2, "delta": 0, "eta": 0},
        ),
        (
            Case(
                "001",
                (0, 0, 1),
                (16.536647016477247, 0, 33.07329403295449, 0, 0, -90),
                12.39842 / 1.650,
            ),
            {"phi": -pi / 2, "chi": 0, "delta": 0},
        ),
    ],
)
def test_three_two_circle_i06_i10_horizontal_fails_for_non_unique_chi(
    hklcalc: HklCalculation, case: Case, constraints: Dict[str, float]
):
    hklcalc.ubcalc.set_lattice("xtal", LatticeParams(5.34, 13.2))
    hklcalc.ubcalc.set_u(I)
    hklcalc.constraints.asdict = constraints

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
def test_i16_cubic_get_hkl(hklcalc: HklCalculation, case: Case):
    u_matrix = np.array(
        (
            (0.9996954135095477, -0.01745240643728364, -0.017449748351250637),
            (0.01744974835125045, 0.9998476951563913, -0.0003045864904520898),
            (0.017452406437283505, -1.1135499981271473e-16, 0.9998476951563912),
        )
    )
    hklcalc.ubcalc.set_lattice("Cubic", LatticeParams(1))
    hklcalc.ubcalc.set_u(u_matrix)

    hkl = hklcalc.get_hkl(Position(*case.position * ureg.deg), 1.0)

    assert np.all(np.round(hkl, 7) == np.round(case.hkl, 7))
