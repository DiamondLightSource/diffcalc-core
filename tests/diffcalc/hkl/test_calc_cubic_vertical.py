import itertools
from math import cos, pi, radians, sin, sqrt
from typing import Dict, Union

import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
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


def test_get_position_with_radians(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"delta": 60, "a_eq_b": True, "mu": 0}))
    case = Case("100", (1, 0, 0), (0, pi / 3, 0, pi / 6, 0, 0))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, asdegrees=False)


@pytest.mark.parametrize(
    ("constraints"),
    [{}, {"mu": 1, "eta": 1, "bisect": True}, {"bisect": True, "eta": 34, "naz": 3}],
)
def test_get_position_raises_exception_if_badly_constrained(
    cubic: UBCalculation, constraints: Dict[str, Union[float, bool]]
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)

    with pytest.raises(DiffcalcException):
        hklcalc.get_position(1, 0, 0, 1)


@pytest.mark.parametrize(
    ("constraints"),
    [
        {"mu": 2, "eta": 2, "delta": 2},
        {"omega": 2, "bisect": True, "delta": 2},
        {"mu": 2, "bisect": True, "delta": 2},
        {"eta": 2, "bisect": True, "delta": 2},
        {"chi": 2, "phi": 2, "delta": 2},
        {"mu": 2, "phi": 2, "delta": 2},
        # {"mu": 2, "chi": 0, "delta": 0},
        {"eta": 2, "phi": 2, "delta": 2},
        {"eta": 2, "chi": 2, "delta": 2},
    ],
)
def test_get_position_for_two_samp_one_det_constraints(
    cubic: UBCalculation, constraints: Dict[str, Union[float, bool]]
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)

    hklcalc.get_position(1, 0, 0, 1)


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


# need one more test to see what happens if ub not configured...


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
            {"delta": 60, "a_eq_b": True, "mu": 0},
            {"a_eq_b": True, "mu": 0, "nu": 0},
            {"psi": 90, "mu": 0, "nu": 0},
            {"a_eq_b": True, "mu": 0, "qaz": 90},
        ],
    ),
)
def test_various_constraints_and_zrots(
    cubic: UBCalculation, case: Case, zrot: float, constraints: Dict[str, float]
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    pos = case.position
    new_position = (*pos[:5], pos[5] + zrot)
    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)

    configure_ub(hklcalc, zrot, 0)
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
            {"psi": 90, "mu": 0, "nu": 0},
            {"a_eq_b": True, "mu": 0, "qaz": 90},
        ],
    ),
)
def test_various_constraints_and_small_zrot_yrot(
    cubic: UBCalculation,
    case: Case,
    zrot: float,
    constraints: Dict[str, float],
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    pos = case.position
    if case.name != "010":
        new_position = (*pos[:4], pos[4] - 2, pos[5] + zrot)
    else:
        new_position = (*pos[:5], pos[5] + zrot)

    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)

    configure_ub(hklcalc, zrot, 2)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


@pytest.mark.parametrize(
    ("constraints"),
    [
        {"a_eq_b": True, "mu": 0, "nu": 0},
        {"psi": 90, "mu": 0, "nu": 0},
        {"a_eq_b": True, "mu": 0, "qaz": 90},
    ],
)
def test_hkl_delta_greater_than_90(cubic: UBCalculation, constraints):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    case = Case("0.1 0 1.5", (0.1, 0, 1.5), (0, 97.469592, 0, 48.734796, 86.185925, 0))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


@pytest.mark.parametrize(
    ("case, zrot, yrot, constraints"),
    itertools.product(
        [
            Case("010", (0, 1, 0), (0, 60, 0, 30, 0, 90)),
        ],
        [1, -1],
        [
            0,
        ],
        [
            {"a_eq_b": True, "mu": 0, "nu": 0},
            {"psi": 90, "mu": 0, "nu": 0},
            {"a_eq_b": True, "mu": 0, "qaz": 90},
        ],
    ),
)
def test_small_zrot(
    cubic: UBCalculation,
    case: Case,
    zrot: float,
    yrot: float,
    constraints: Dict[str, float],
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    pos = case.position
    new_position = (*pos[:4], pos[4] - yrot, pos[5] + zrot)
    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)

    configure_ub(hklcalc, zrot, yrot)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


@pytest.mark.parametrize(
    ("zrot", "constraints"),
    itertools.product(
        [0, 2, -2, 45, -45, 90, -90],
        [
            {"delta": 60, "a_eq_b": True, "mu": 0},
            {"a_eq_b": True, "mu": 0, "nu": 0},
            {"psi": 90, "mu": 0, "nu": 0},
            {"a_eq_b": True, "mu": 0, "qaz": 90},
        ],
    ),
)
def test_fails_for_parallel_vectors(
    cubic: UBCalculation, zrot: float, constraints: Dict[str, Union[float, bool]]
):
    """Confirm that a hkl of (0,0,1) fails for a_eq_b=True.

    By default, the reference vector is (0,0,1). A parallel vector to this should
    cause failure
    """
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0 + zrot))

    configure_ub(hklcalc, zrot, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("constraints"),
    [
        {"qaz": 90, "alpha": 90, "phi": 0},
        {"delta": 90, "beta": 0, "phi": 0},
        {"delta": 90, "betain": 0, "phi": 0},
    ],
)
def test_alpha_90(cubic: UBCalculation, constraints: Dict[str, float]):
    hklcalc = HklCalculation(cubic, Constraints(constraints))
    hklcalc.ubcalc.n_hkl = (1, -1, 0)

    case = Case("sqrt(2)00", (sqrt(2), 0, 0), (0, 90, 0, 45, 0, 0))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


def test_ttheta_180(cubic: UBCalculation):
    cubic.n_hkl = (1, -1, 0)  # type: ignore
    hklcalc = HklCalculation(cubic, Constraints({"nu": 0, "chi": 0, "phi": 0}))
    case = Case("200", (2, 0, 0), (0, 180, 0, 90, 0, 0))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
def test_with_chi_phi_constrained(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"nu": 0, "chi": 90.0, "phi": 0.0}))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
def test_fixed_phi_0(cubic: UBCalculation, case: Case, yrot: float):
    hklcalc = HklCalculation(cubic, Constraints({"mu": 0, "nu": 0, "phi": 0}))

    pos = case.position
    new_position = (*pos[:3], pos[3], pos[4] - yrot, pos[5])

    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)

    configure_ub(hklcalc, 0, yrot)
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
def test_fixed_phi_30(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"mu": 0, "nu": 0, "phi": 30}))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("yrot"),
    [0, 2, -2, 45, -45, 90, -90],
)
def test_fixed_phi_90(cubic: UBCalculation, yrot: float):
    hklcalc = HklCalculation(cubic, Constraints({"mu": 0, "nu": 0, "phi": 90}))

    case = Case("010", (0, 1, 0), (0, 60, 0, 30, 0, 90))

    configure_ub(hklcalc, 0, yrot)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
def test_fixed_mu_eta(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"nu": 0, "mu": 90, "eta": 0}))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


def test_fixed_mu_eta_fails_for_non_unique_phi(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"nu": 0, "mu": 90, "eta": 0}))

    case = Case("001", (0, 0, 1), (90, 60, 0, 0, 60, 0))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case"),
    (
        Case(
            "100",
            (1, 0, 0),
            (0, 60, 0, 30, 0, 0),
        ),
        Case(
            "010-->100",
            (sin(radians(4)), cos(radians(4)), 0),
            (0, 60, 0, 120 - 4, 0, 0),
        ),
        Case("010", (0, 1, 0), (0, 60, 0, 120, 0, 0)),
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
            (0, 60, 0, 30, 90 - 4, 0),
        ),
    ),
)
def test_with_one_ref_and_mu_0_phi_0(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"psi": 90, "mu": 0, "phi": 0}))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


def test_with_one_ref_and_mu_0_phi_0_fails_for_parallel_vectors(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"psi": 90, "mu": 0, "phi": 0}))

    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case"),
    (
        Case(
            "100",
            (1, 0, 0),
            (-90, 60, 0, 0, 30, 0),
        ),
        Case(
            "100-->001",
            (cos(radians(4)), 0, sin(radians(4))),
            (-90, 60, 0, 0, 30 + 4, 0),
        ),
        Case("010", (0, 1, 0), (120, 0, 60, 0, 180, 0)),
        Case(
            "0.1 0 0.15",
            (0.1, 0, 0.15),
            (
                -90,
                10.34318,
                0,
                0,
                61.48152,
                0,
            ),
        ),
        Case(
            "010-->001",
            (0, cos(radians(4)), sin(radians(4))),
            (120 + 4, 0, 60, 0, 180, 0),
        ),
    ),
)
def test_with_one_ref_and_eta_0_phi_0(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"psi": 0, "eta": 0, "phi": 0}))
    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


def test_with_one_ref_and_eta_0_phi_0_fails_for_parallel_vectors(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"psi": 0, "eta": 0, "phi": 0}))

    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0))
    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
def test_with_bisect_and_nu_0_omega_0(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"bisect": True, "nu": 0, "omega": 0}))
    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


def test_with_bisect_and_nu_0_omega_0_fails_for_non_unique_phi(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"bisect": True, "nu": 0, "omega": 0}))

    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0))
    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
            {"a_eq_b": True, "qaz": 90, "chi": 90},
            {"a_eq_b": True, "nu": 0, "chi": 90},
        ],
    ),
)
def test_det_ref_and_chi_constrained(
    cubic: UBCalculation, case: Case, constraints: Dict[str, Union[float, bool]]
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case", "constraints"),
    itertools.product(
        [
            Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0)),
        ],
        [
            {"a_eq_b": True, "qaz": 90, "chi": 90},
            {"a_eq_b": True, "nu": 0, "chi": 90},
        ],
    ),
)
def test_det_ref_and_chi_constrained_fails_for_parallel_vectors(
    cubic: UBCalculation, case: Case, constraints: Dict[str, Union[float, bool]]
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
            {"nu": 60, "psi": 0, "phi": 0},
        ],
    ),
)
def test_ref_and_fixed_delta_phi_0(
    cubic: UBCalculation,
    case: Case,
    zrot: float,
    yrot: float,
    constraints: Dict[str, float],
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

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

    configure_ub(hklcalc, zrot, yrot)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


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
def test_fixed_delta_eta_phi_0(cubic: UBCalculation, case: Case, yrot: float):
    hklcalc = HklCalculation(cubic, Constraints({"eta": 0, "delta": 0, "phi": 0}))

    new_position = (*case.position[:4], case.position[4] - yrot, case.position[5])
    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)

    configure_ub(hklcalc, 0, yrot)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


def test_fixed_delta_eta_phi_0_fails_as_non_unique_chi(cubic: UBCalculation):
    case = Case("010", (0, 1, 0), (120, 0, 60, 0, 0, 0))

    hklcalc = HklCalculation(cubic, Constraints({"eta": 0, "delta": 0, "phi": 0}))
    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case"),
    (
        Case("100", (1, 0, 0), (0, 0, 60, 0, -90, 30)),
        Case("010", (0, 1, 0), (90, 0, 60, 0, -90, 30)),
        Case("001", (0, 0, 1), (30, 0, 60, 0, 0, 30)),
    ),
)
def test_fixed_delta_eta_phi_30(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"eta": 0, "delta": 0, "phi": 30}))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
def test_fixed_delta_eta_chi_0(cubic: UBCalculation, case: Case, zrot: float):
    hklcalc = HklCalculation(cubic, Constraints({"eta": 0, "delta": 0, "chi": 0}))

    new_position = (*case.position[:5], case.position[5] + zrot)
    case_with_rotation = Case(case.name, case.hkl, new_position, case.wavelength)
    configure_ub(hklcalc, zrot, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


# phi || q
def test_fixed_delta_eta_chi_0_fails_for_degenerate_case(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"eta": 0, "delta": 0, "chi": 0}))

    case = Case("001", (0, 0, 1), (30, 0, 60, 0, 0, 0))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


# Line 2137... 22 tests to go!


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
def test_fixed_delta_eta_chi_30(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"eta": 0, "delta": 0, "chi": 30}))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
def test_fixed_gamma_mu_chi_30(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"mu": 0, "nu": 0, "chi": 30}))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


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
def test_fixed_gamma_mu_chi_90(cubic: UBCalculation, case: Case, zrot: float):
    hklcalc = HklCalculation(cubic, Constraints({"mu": 0, "nu": 0, "chi": 90}))

    case_with_rotation = Case(
        case.name, case.hkl, (*case.position[:5], case.position[5] + zrot)
    )

    configure_ub(hklcalc, zrot, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case_with_rotation)


def test_fixed_gamma_mu_chi_90_fails_for_non_unique_phi(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"mu": 0, "nu": 0, "chi": 90}))

    case = Case("001", (0, 0, 1), (0, 60, 0, 30, 90, 0))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case"),
    (
        Case("100", (1, 0, 0), (-90, 60, 0, 0, 30, 0)),
        Case("010", (0, 1, 0), (90, 60, 0, 0, 150, -90)),
        Case("110", (1, 1, 0), (-90, 90, 0, 0, 45, 45)),
        Case("111", (1, 1, 1), (90, 120, 0, 0, 84.7356, -135)),
    ),
)
def test_fixed_naz_psi_eta(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"naz": 90, "psi": 0, "eta": 0}))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


def test_fixed_naz_psi_eta_fails_for_parallel_vectors(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"naz": 90, "psi": 0, "eta": 0}))

    case = Case("001", (0, 0, 1), (30, 0, 60, 90, 0, 0))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


@pytest.mark.parametrize(
    ("case"),
    (
        Case("010", (0, 1, 0), (120, 0, 60, 0, 90, 0)),
        Case("011", (0, 1, 1), (45, 0, -90, 135, 90, 0)),
        Case("100", (1, 0, 0), (-30, 0, -60, 0, 90, 0)),
        Case("101", (1, 0, 1), (-30, 54.7356, -90, 35.2644, 90, 0)),
        Case("1 1 0.0001", (1, 1, 0.0001), (0, -0.00286, -90, 179.997, 90, 0)),
        Case("111", (1, 1, 1), (-171.5789, 20.9410, 122.3684, -30.3612, 90, 0)),
        Case("1.1 0 0", (1.1, 0, 0), (-146.6330, 0, 66.7340, 0, 90, 0)),
        Case("0.9 0 0", (0.9, 0, 0), (-153.2563, 0, 53.4874, 0, 90, 0)),
        Case(
            "0.7 0.8 0.8",
            (0.7, 0.8, 0.8),
            (167.7652, 23.7336, 82.7832, -24.1606, 90, 0),
        ),
        Case(
            "0.7 0.8 0.9",
            (0.7, 0.8, 0.9),
            (169.0428, 25.6713, 88.0926, -27.2811, 90, 0),
        ),
        Case("0.7 0.8 1", (0.7, 0.8, 1), (170.5280, 27.1595, 94.1895, -30.4583, 90, 0)),
    ),
)
def test_fixed_chi_phi_a_eq_b(cubic: UBCalculation, case: Case):
    hklcalc = HklCalculation(cubic, Constraints({"chi": 90, "phi": 0, "a_eq_b": True}))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


def test_fixed_chi_phi_a_eq_b_fails_for_non_unique_eta(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"chi": 90, "phi": 0, "a_eq_b": True}))

    case = Case("1m10", (1, -1, 0), (-90, 0, 90, 0, 90, 0))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


def test_fixed_chi_phi_a_eq_b_fails_for_parallel_vector(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"chi": 90, "phi": 0, "a_eq_b": True}))

    case = Case("001", (0, 0, 1), (30, 0, 60, 90, 0, 0))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


@pytest.mark.parametrize(
    ("case", "constraints"),
    [
        (Case("001", (0, 0, 1), (30, 0, 60, 0, 0, 0)), {"chi": 0, "phi": 0, "eta": 0}),
        (Case("010", (0, 1, 0), (120, 0, 60, 0, 0, 0)), {"chi": 0, "phi": 0, "eta": 0}),
        (Case("011", (0, 1, 1), (90, 0, 90, 0, 0, 0)), {"chi": 0, "phi": 0, "eta": 0}),
        (Case("100", (1, 0, 0), (0, 60, 0, 0, 0, 30)), {"chi": 0, "phi": 30, "eta": 0}),
        (Case("100", (1, 0, 0), (0, 60, 0, 30, 0, 0)), {"chi": 0, "phi": 0, "eta": 30}),
        (Case("110", (1, 1, 0), (0, 90, 0, 0, 0, 90)), {"chi": 0, "phi": 90, "eta": 0}),
        (Case("110", (1, 1, 0), (0, 90, 0, 90, 0, 0)), {"chi": 0, "phi": 0, "eta": 90}),
        (
            Case("0.01 0.01 0.1", (0.01, 0.01, 0.1), (8.6194, 0.5730, 5.7607, 0, 0, 0)),
            {"chi": 0, "phi": 0, "eta": 0},
        ),
        (
            Case("0 0 0.1", (0, 0, 0.1), (2.8660, 0, 5.7320, 0, 0, 0)),
            {"chi": 0, "phi": 0, "eta": 0},
        ),
        (
            Case("0.1 0 0.01", (0.1, 0, 0.01), (30.3314, 5.7392, 0.4970, 0, 0, 0)),
            {"chi": 0, "phi": 0, "eta": 0},
        ),
        (
            Case("0.1 0 0.01", (0.1, 0, 0.01), (30.3314, 5.7392, 0.4970, 0, 0, 0)),
            {"chi": 0, "phi": 0, "eta": 0},
        ),
        (
            Case("", (0, cos(radians(4)), sin(radians(4))), (120 - 4, 0, 60, 0, 0, 0)),
            {"chi": 0, "phi": 0, "eta": 0},
        ),
        (
            Case("001", (0, 0, 1), (2.8660, 0, 5.7320, 0, 0, 0), 0.1),
            {"chi": 0, "phi": 0, "eta": 0},
        ),
        (
            Case("001", (0, 0, 1), (2.8660, 0, 5.7320, 0, 0, 0), 0.1),
            {"chi": 0, "phi": 0, "eta": 0},
        ),
        (
            Case("1 0 0.1", (1, 0, 0.1), (30.3314, 5.7392, 0.4970, 0, 0, 0), 0.1),
            {"chi": 0, "phi": 0, "eta": 0},
        ),
    ],
)
def test_chi_phi_eta(cubic: UBCalculation, case: Case, constraints: Dict[str, float]):
    hklcalc = HklCalculation(cubic, Constraints(constraints))
    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


@pytest.mark.parametrize(
    ("case", "constraints"),
    [
        (Case("", (0, 0, 1), (0, 60, 0, 30, 90, 0)), {"chi": 90, "phi": 0, "mu": 0}),
        (Case("", (0, 1, 0), (0, 60, 0, 120, 90, 0)), {"chi": 90, "phi": 0, "mu": 0}),
        (Case("", (0, 1, 1), (0, 90, 0, 90, 90, 0)), {"chi": 90, "phi": 0, "mu": 0}),
        (Case("", (1, 0, 0), (0, 60, 0, -60, 90, 90)), {"chi": 90, "phi": 90, "mu": 0}),
        (Case("", (1, 0, 1), (0, 90, 0, 0, 90, 90)), {"chi": 90, "phi": 90, "mu": 0}),
        (
            Case("", (sin(radians(4)), cos(radians(4)), 0), (0, 60, 0, 120 - 4, 0, 0)),
            {"chi": 0, "phi": 0, "mu": 0},
        ),
    ],
)
def test_mu_chi_phi(cubic: UBCalculation, case: Case, constraints: Dict[str, float]):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


def test_mu_chi_phi_fails_as_non_unique_sample_orientation(cubic: UBCalculation):
    hklcalc = HklCalculation(cubic, Constraints({"chi": 90, "phi": 0, "mu": 90}))

    case = Case("", (-1, 1, 0), (90, 90, 0, 90, 90, 0))
    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


@pytest.mark.parametrize(
    ("case", "constraints"),
    [
        (Case("", (0, 1, 0), (0, 60, 0, 120, 90, 0)), {"chi": 90, "eta": 120, "mu": 0}),
        (
            Case("", (1, 0, 0), (0, 60, 0, -60, 90, 90)),
            {"chi": 90, "eta": -60, "mu": 0},
        ),
        (
            Case("", (-1, 1, 0), (90, 90, 0, 90, 90, 0)),
            {"chi": 90, "eta": 90, "mu": 90},
        ),
        (Case("", (1, 0, 1), (0, 90, 0, 0, 90, 90)), {"chi": 90, "eta": 0, "mu": 0}),
        (
            Case("", (sin(radians(4)), cos(radians(4)), 0), (0, 60, 0, 0, 0, 120 - 4)),
            {"chi": 0, "eta": 0, "mu": 0},
        ),
    ],
)
def test_mu_eta_chi(cubic: UBCalculation, case: Case, constraints: Dict[str, float]):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


@pytest.mark.parametrize(
    ("case", "constraints"),
    (
        (Case("", (0, 1, 1), (0, 90, 0, 90, 90, 0)), {"chi": 90, "eta": 90, "mu": 0}),
        (Case("", (0, 0, 1), (0, 60, 0, 30, 90, 0)), {"eta": 30, "chi": 90, "mu": 0}),
    ),
)
def test_mu_eta_chi_fails_as_non_unique_sample_orientation(
    cubic: UBCalculation, case: Case, constraints: Dict[str, float]
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


@pytest.mark.parametrize(
    ("case", "constraints"),
    [
        (Case("", (0, 0, 1), (30, 0, 60, 0, 0, 0)), {"eta": 0, "phi": 0, "mu": 30}),
        (Case("", (0, 1, 1), (0, 90, 0, 90, 90, 0)), {"eta": 90, "phi": 0, "mu": 0}),
        (
            Case("", (-1, 1, 0), (90, 90, 0, 90, 90, 0)),
            {"eta": 90, "phi": 0, "mu": 90},
        ),
        (
            Case("", (sin(radians(4)), 0, cos(radians(4))), (0, 60, 0, 30, 90 - 4, 0)),
            {"eta": 30, "phi": 0, "mu": 0},
        ),
    ],
)
def test_mu_eta_phi(cubic: UBCalculation, case: Case, constraints: Dict[str, float]):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


@pytest.mark.parametrize(
    ("case", "constraints"),
    [
        (Case("", (0, 1, 0), (0, 60, 0, 120, 90, 0)), {"eta": 120, "phi": 0, "mu": 0}),
        (
            Case("", (1, 0, 0), (0, 60, 0, -60, 90, 90)),
            {"eta": -60, "phi": 90, "mu": 0},
        ),
        (Case("", (1, 0, 1), (0, 90, 0, 0, 90, 90)), {"eta": 0, "phi": 90, "mu": 0}),
    ],
)
def test_mu_eta_phi_fails_as_non_unique_sample_orientation(
    cubic: UBCalculation, case: Case, constraints: Dict[str, float]
):
    hklcalc = HklCalculation(cubic, Constraints(constraints))

    configure_ub(hklcalc, 0, 0)
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)
