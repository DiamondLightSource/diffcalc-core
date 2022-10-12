import itertools
from typing import Dict, Union

import numpy as np
import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import DiffcalcException, I

from tests.diffcalc.hkl.test_calc import (
    Case,
    convert_position_to_hkl_and_hkl_to_position,
)


@pytest.fixture
def hklcalc() -> HklCalculation:
    ubcalc = UBCalculation()
    ubcalc.n_phi = (0, 0, 1)  # type: ignore
    ubcalc.surf_nphi = (0, 0, 1)  # type: ignore
    return HklCalculation(ubcalc, Constraints())


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
        ),  # this one shouldn't pass!!!!
        Case(
            "101",
            (1, 0, 1),
            (0, 98.74666191021282, 0, 117.347760720783, 90, -90),
            12.39842 / 1.650,
        ),
    ],
)
def test_three_two_circle_i06_i10(hklcalc: HklCalculation, case: Case):
    hklcalc.ubcalc.set_lattice("xtal", 5.34, 13.2)
    hklcalc.ubcalc.set_u(I)
    hklcalc.constraints.asdict = {"phi": -90, "nu": 0, "mu": 0}

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
            {"phi": -90, "delta": 0, "eta": 0},
        ),
        (
            Case(
                "101",
                (1, 0, 1),
                (117.347760720783, 0, 98.74666191021282, 0, 0, -90),
                12.39842 / 1.650,
            ),
            {"phi": -90, "delta": 0, "eta": 0},
        ),
        (
            Case(
                "100",
                (1, 0, 0),
                (134.71463281804702, 0, 89.42926563609406, 0, 0, -90),
                12.39842 / 1.650,
            ),
            {"phi": -90, "chi": 0, "delta": 0},
        ),
    ],
)
def test_three_two_circle_i06_i10_horizontal(
    hklcalc: HklCalculation, case: Case, constraints: Dict[str, float]
):
    hklcalc.ubcalc.set_lattice("xtal", 5.34, 13.2)
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
            {"phi": -90, "delta": 0, "eta": 0},
        ),
        (
            Case(
                "001",
                (0, 0, 1),
                (16.536647016477247, 0, 33.07329403295449, 0, 0, -90),
                12.39842 / 1.650,
            ),
            {"phi": -90, "chi": 0, "delta": 0},
        ),
    ],
)
def test_three_two_circle_i06_i10_horizontal_fails_for_non_unique_chi(
    hklcalc: HklCalculation, case: Case, constraints: Dict[str, float]
):
    hklcalc.ubcalc.set_lattice("xtal", 5.34, 13.2)
    hklcalc.ubcalc.set_u(I)
    hklcalc.constraints.asdict = constraints

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case", "places"),
    [
        (Case("010", (0, 1, 0), (0, 60, 0, 120, 0, 0)), 4),
        (Case("011", (0, 1, 1), (30, 54.7356, 90, 125.2644, 0, 0)), 4),
        (Case("100", (1, 0, 0), (0, 60, 0, 30, 0, 0)), 4),
        (Case("101", (1, 0, 1), (30, 54.7356, 90, 35.2644, 0, 0)), 4),
        (Case("110", (1, 1, 0), (0, 90, 0, 90, 0, 0)), 4),
        (Case("1 1 0.0001", (1, 1, 0.0001), (0.0029, 89.9971, 90.0058, 90, 0, 0)), 3),
        (Case("111", (1, 1, 1), (30, 54.7356, 150, 99.7356, 0, 0)), 4),
        (Case("1.1 0 0", (1.1, 0, 0), (0, 66.7340, 0, 33.3670, 0, 0)), 4),
        (Case("0.9 0 0", (0.9, 0, 0), (0, 53.4874, 0, 26.7437, 0, 0)), 4),
        (
            Case(
                "0.7 0.8 0.8",
                (0.7, 0.8, 0.8),
                (23.5782, 59.9980, 76.7037, 84.2591, 0, 0),
            ),
            4,
        ),
        (
            Case(
                "0.7 0.8 0.9",
                (0.7, 0.8, 0.9),
                (26.74368, 58.6754, 86.6919, 85.3391, 0, 0),
            ),
            4,
        ),
        (Case("0.7 0.8 1", (0.7, 0.8, 1), (30, 57.0626, 96.86590, 86.6739, 0, 0)), 4),
    ],
)
def test_i07_horizontal(hklcalc: HklCalculation, case: Case, places: int):
    hklcalc.ubcalc.set_lattice("Cubic", 1)
    hklcalc.constraints.asdict = {"chi": 0, "phi": 0, "a_eq_b": True}
    hklcalc.ubcalc.set_u(I)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, places)


def test_i07_horizontal_fails_for_parallel_vectors(hklcalc: HklCalculation):
    hklcalc.ubcalc.set_lattice("Cubic", 1)
    hklcalc.constraints.asdict = {"chi": 0, "phi": 0, "a_eq_b": True}
    hklcalc.ubcalc.set_u(I)

    case = Case("", (0, 0, 1), (30, 0, 60, 90, 0, 0))

    with pytest.raises(DiffcalcException):
        convert_position_to_hkl_and_hkl_to_position(hklcalc, case)


@pytest.mark.parametrize(
    ("case"),
    [
        Case("", (1, 1, 0.001), (-89.9714, 90.0430, -89.9618, 90.0143, 90, 0)),
        Case("", (1, 1, 0.1), (-87.1331, 85.6995, 93.8232, 91.4339, 90, 0)),
        Case("", (1, 1, 0.5), (-75.3995, 68.0801, 109.5630, 97.3603, 90, 0)),
        Case("", (1, 1, 1), (-58.6003, 42.7342, 132.9005, 106.3250, 90, 0)),
    ],
)
def test_i16_vertical(hklcalc: HklCalculation, case: Case):
    hklcalc.ubcalc.set_lattice("Cubic", 1)
    hklcalc.constraints.asdict = {"chi": 90, "psi": 90, "phi": 0}
    hklcalc.ubcalc.set_u(I)

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 4)


@pytest.mark.parametrize(
    ("case", "constraints"),
    itertools.product(
        [
            Case(
                "",
                (-1.1812112493619709, -0.71251524866987204, 5.1997083010199221),
                (26, 0, 52, 0, 45.2453, 186.6933 - 360),
                12.39842 / 8,
            )
        ],
        [
            {"delta": 0, "a_eq_b": True, "eta": 0},
            {"delta": 0, "psi": -90, "eta": 0},
            {"delta": 0, "alpha": 17.9776, "eta": 0},
        ],
    ),
)
def test_i16_failed_hexagonal_experiment(
    hklcalc: HklCalculation, case: Case, constraints: Dict[str, Union[float, bool]]
):
    hklcalc.ubcalc.set_lattice("I16_test", "Hexagonal", 4.785, 12.991)
    u_matrix = np.array(
        [
            [-9.65616334e-01, -2.59922060e-01, 5.06142415e-03],
            [2.59918682e-01, -9.65629598e-01, -1.32559487e-03],
            [5.23201232e-03, 3.55426382e-05, 9.99986312e-01],
        ]
    )
    hklcalc.ubcalc.set_u(u_matrix)

    hklcalc.constraints.asdict = constraints

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 3)


def test_i16_failed_hexagonal_experiment_with_small_variations(hklcalc: HklCalculation):
    hklcalc.ubcalc.set_lattice("I16_test", "Hexagonal", 4.785, 12.991)
    u_matrix = np.array(
        [
            [-9.65616334e-01, -2.59922060e-01, 5.06142415e-03],
            [2.59918682e-01, -9.65629598e-01, -1.32559487e-03],
            [5.23201232e-03, 3.55426382e-05, 9.99986312e-01],
        ]
    )
    hklcalc.ubcalc.set_u(u_matrix)
    hklcalc.constraints.asdict = {"delta": 0, "alpha": 17.8776, "eta": 0}

    case = Case(
        "",
        (-1.1812112493619709, -0.71251524866987204, 5.1997083010199221),
        (25.85, 0, 52, 0, 45.2453, -173.518),
        12.39842 / 8,
    )
    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 3)


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
    hklcalc.ubcalc.set_lattice("Cubic", 1)
    hklcalc.ubcalc.set_u(u_matrix)

    hkl = hklcalc.get_hkl(Position(*case.position), 1.0)

    assert np.all(np.round(hkl, 7) == np.round(case.hkl, 7))


@pytest.mark.parametrize(
    ("case"),
    [
        Case("", (2, 0, 1e-6), (0, 20.377277, 0, 10.188639, 0.000029, 0)),
        Case("", (2, 1e-6, 0), (0, 20.377277, 0, 10.188667, 0, 0)),
    ],
)
def test_i16_cubic(hklcalc: HklCalculation, case: Case):
    hklcalc.ubcalc.set_lattice("xtal", 5.653244295348863)
    hklcalc.ubcalc.set_u(I)
    hklcalc.ubcalc.n_phi = (0, 0, 1)

    hklcalc.constraints.asdict = {"mu": 0, "nu": 0, "phi": 0}

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 6)


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
    hklcalc.constraints.asdict = {"qaz": 90, "alpha": 11, "mu": 0}

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

    hklcalc.constraints.asdict = {"psi": 10, "mu": 0, "nu": 0}

    convert_position_to_hkl_and_hkl_to_position(hklcalc, case, 3)


# Test_I21ExamplesUB
