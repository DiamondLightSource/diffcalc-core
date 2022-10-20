from math import cos, radians, sin
from typing import Dict, Tuple, Union

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
def cubic() -> HklCalculation:
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore

    ubcalc.set_lattice("Cubic", 1.0)
    configure_ub(ubcalc)

    return HklCalculation(ubcalc, Constraints())


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

    assert tuple(
        [item.magnitude for item in all_positions[0][0].astuple]
    ) == pytest.approx(expected_position)


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
def test_mu_eta_phi(cubic: HklCalculation, case: Case, constraints: Dict[str, float]):
    cubic.constraints = Constraints(constraints)

    convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


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
    cubic: HklCalculation, case: Case, constraints: Dict[str, float]
):
    cubic.constraints = Constraints(constraints)

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


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
def test_chi_phi_eta(cubic: HklCalculation, case: Case, constraints: Dict[str, float]):
    cubic.constraints = Constraints(constraints)
    convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


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
def test_mu_chi_phi(cubic: HklCalculation, case: Case, constraints: Dict[str, float]):
    cubic.constraints = Constraints(constraints)

    convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


def test_mu_chi_phi_fails_as_non_unique_sample_orientation(cubic: HklCalculation):
    cubic.constraints = Constraints({"chi": 90, "phi": 0, "mu": 90})

    case = Case("", (-1, 1, 0), (90, 90, 0, 90, 90, 0))
    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


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
def test_mu_eta_chi(cubic: HklCalculation, case: Case, constraints: Dict[str, float]):
    cubic.constraints = Constraints(constraints)

    convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)


@pytest.mark.parametrize(
    ("case", "constraints"),
    (
        (Case("", (0, 1, 1), (0, 90, 0, 90, 90, 0)), {"chi": 90, "eta": 90, "mu": 0}),
        (Case("", (0, 0, 1), (0, 60, 0, 30, 90, 0)), {"eta": 30, "chi": 90, "mu": 0}),
    ),
)
def test_mu_eta_chi_fails_as_non_unique_sample_orientation(
    cubic: HklCalculation, case: Case, constraints: Dict[str, float]
):
    cubic.constraints = Constraints(constraints)

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(cubic, case, 4)
