from math import pi
from typing import Dict, List, Optional, Union

import numpy as np
import pytest
from diffcalc.hkl.constraints import CATEGORY, TYPE, Constraint, Constraints
from diffcalc.util import DiffcalcException
from typing_extensions import Literal

from tests.test_tools import eq_
from tests.tools import assert_dict_almost_equal


@pytest.fixture
def cm() -> Constraints:
    return Constraints()


@pytest.mark.parametrize(
    ("cons", "det", "ref", "samp"),
    [
        ({}, {}, {}, {}),
        ({"nu": 0, "mu": 0, "a_eq_b": True}, {"nu": 0}, {"a_eq_b": True}, {"mu": 0}),
    ],
)
def test_init_generates_correct_dictionaries(
    cons: Dict[str, Union[float, Literal["True"]]],
    det: Dict[str, Union[float, Literal["True"]]],
    ref: Dict[str, Union[float, Literal["True"]]],
    samp: Dict[str, Union[float, Literal["True"]]],
):
    constraints = Constraints(cons)
    assert constraints.asdict == cons
    assert constraints.detector == det
    assert constraints.reference == ref
    assert constraints.sample == samp


@pytest.mark.parametrize(
    ("con_name"),
    [
        "delta",
        "nu",
        "qaz",
        "naz",
        "a_eq_b",
        "alpha",
        "beta",
        "psi",
        "bin_eq_bout",
        "betain",
        "betaout",
        "mu",
        "eta",
        "chi",
        "phi",
        "bisect",
        "omega",
    ],
)
def test_setters_getters_and_deleters_for_each_constraint(
    cm: Constraints, con_name: str
):
    constraint_def: Constraint = getattr(cm, "_" + con_name)

    value: Union[Literal["True"], int] = 1

    if constraint_def.type == TYPE.VOID:
        value = True

    setattr(cm, con_name, value)
    assert getattr(cm, con_name) == value

    delattr(cm, con_name)
    assert getattr(cm, con_name) is None


@pytest.mark.parametrize(
    ("con_dict"),
    [{"nu": 0, "mu": 0, "a_eq_b": None}, {"nu": 3, "mu": 4, "a_eq_b": True}],
)
def test_as_dict_setter_and_getter(
    cm: Constraints, con_dict: Dict[str, Union[float, Literal["True"]]]
):
    non_null_con_dict = {k: v for k, v in con_dict.items() if v is not None}
    cm.asdict = con_dict

    assert {
        key: np.round(value, 8) if isinstance(value, float) else value
        for key, value in cm.asdict.items()
    } == non_null_con_dict


@pytest.mark.parametrize(
    ("con_dict", "con_set", "con_list", "con_tuple"),
    [
        (
            {"nu": 3.0, "mu": 4.0, "a_eq_b": True},
            (("nu", 3.0), "a_eq_b", ("mu", 4)),
            {("nu", 3.0), ("mu", 4), "a_eq_b"},
            [("nu", 3.0), "a_eq_b", ("mu", 4)],
        ),
    ],
)
def test_all_init(con_dict, con_tuple, con_set, con_list):
    for el in (con_dict, con_tuple, con_set, con_list):
        cm = Constraints(el)
        assert_dict_almost_equal(cm.asdict, con_dict)
        # eq_(cm.astuple, con_tuple)


def test_wrong_parameter_to_init_raises_exception():
    with pytest.raises(DiffcalcException):
        Constraints(12)


def test_constraints_stored_as_radians():
    deg_cons = Constraints({"alpha": 90, "mu": 30, "nu": 45})

    assert deg_cons._alpha.value == pi / 2
    assert deg_cons._mu.value == pi / 6
    assert deg_cons._nu.value == pi / 4


def test_str_constraint(cm):
    print(str(cm))
    eq_(
        str(cm),
        "\n".join(
            [
                "    DET             REF             SAMP",
                "    -----------     -----------     -----------",
                "    delta           a_eq_b          mu",
                "    nu              alpha           eta",
                "    qaz             beta            chi",
                "    naz             psi             phi",
                "                    bin_eq_bout     bisect",
                "                    betain          omega",
                "                    betaout",
                "",
                "!   3 more constraints required",
                "",
            ],
        ),
    )


def test_str_constraint_for_non_implemented_combination():
    cons = Constraints({"bisect": True, "omega": 30, "bin_eq_bout": True})

    assert (
        str(cons).split("\n")[-1]
        == "    Sorry, this constraint combination is not implemented."
    )


def test_build_display_table(cm):
    cm.qaz = 1.234
    cm.alpha = 1.0
    cm.eta = 99.0
    print("\n".join(cm._build_display_table_lines()))
    eq_(
        cm._build_display_table_lines(),
        [
            "    DET             REF             SAMP",
            "    -----------     -----------     -----------",
            "    delta           a_eq_b          mu",
            "    nu          --> alpha       --> eta",
            "--> qaz             beta            chi",
            "    naz             psi             phi",
            "                    bin_eq_bout     bisect",
            "                    betain          omega",
            "                    betaout",
        ],
    )


def test_setting_one_constraint_as_none(cm: Constraints):
    cm.asdict = {"delta": 1.0, "mu": 2.0}

    cm.mu = None

    assert cm.asdict == {"delta": 1.0}


def test_setting_invalid_constraint_name(cm: Constraints):
    with pytest.raises(DiffcalcException):
        cm.asdict = {"non_existent": 10}

    with pytest.raises(DiffcalcException):
        cm.astuple = (("non_existent", 10.0),)


def test_constrained_all(cm: Constraints):
    all_constraints: Dict[str, Optional[float]] = {
        "delta": None,
        "nu": None,
        "qaz": None,
        "naz": None,
        "a_eq_b": None,
        "alpha": None,
        "beta": None,
        "psi": None,
        "bin_eq_bout": None,
        "betain": None,
        "betaout": None,
        "mu": None,
        "eta": None,
        "chi": None,
        "phi": None,
        "bisect": None,
        "omega": None,
    }

    assert cm.constrained() == set()
    assert cm.all() == all_constraints

    cm.delta = 10.0
    cm.alpha = 20.0
    cm.mu = 30.0
    all_constraints.update({"delta": 10.0, "alpha": 20.0, "mu": 30.0})

    assert cm.constrained() == {cm._delta, cm._alpha, cm._mu}
    assert cm.constrained(category=CATEGORY.DETECTOR) == {cm._delta}
    assert cm.constrained(category=CATEGORY.REFERENCE) == {cm._alpha}
    assert cm.constrained(category=CATEGORY.SAMPLE) == {cm._mu}
    assert_dict_almost_equal(cm.all(), all_constraints)
    assert_dict_almost_equal(
        cm.all(category=CATEGORY.DETECTOR),
        {
            "delta": 10.0,
            "nu": None,
            "qaz": None,
            "naz": None,
        },
    )
    assert_dict_almost_equal(
        cm.all(category=CATEGORY.REFERENCE),
        {
            "a_eq_b": None,
            "alpha": 20.0,
            "beta": None,
            "psi": None,
            "bin_eq_bout": None,
            "betain": None,
            "betaout": None,
        },
    )
    assert_dict_almost_equal(
        cm.all(category=CATEGORY.SAMPLE),
        {
            "mu": 30.0,
            "eta": None,
            "chi": None,
            "phi": None,
            "bisect": None,
            "omega": None,
        },
    )


def test_clear_constraints(cm):
    cm.asdict = {"delta": 1.0, "mu": 2.0}

    cm.clear()
    assert cm.asdict == {}


@pytest.mark.parametrize(
    ("starting_constraints"),
    [
        {"alpha": 1},
        {"mu": 1},
        {"alpha": 1, "mu": 1},
        {"mu": 1, "eta": 1},
        {"alpha": 1, "mu": 1, "eta": 1},
        {"mu": 1, "eta": 1, "chi": 1},
        {"delta": 1, "eta": 1, "chi": 1},
    ],
)
def test_adding_detector_constraint_to_existing_set_of_constraints(
    starting_constraints: Dict[str, Union[float, Literal["True"]]],
):
    cons = Constraints(starting_constraints)

    extra_constraints = {"delta": 1, "naz": 2}

    if (len(cons.asdict) == 3) & (len(cons.detector) == 0):
        with pytest.raises(DiffcalcException):
            cons.asdict = {**cons.asdict, **extra_constraints}
        return

    cons.asdict = {**cons.asdict, **extra_constraints}

    assert pytest.approx(cons.naz) == 2
    assert cons.delta is None


@pytest.mark.parametrize(
    ("starting_constraints"),
    [
        {"delta": 1},
        {"mu": 1},
        {"delta": 1, "mu": 1},
        {"mu": 1, "eta": 1},
        {"delta": 1, "mu": 1, "eta": 1},
        {"mu": 1, "eta": 1, "chi": 1},
        {"a_eq_b": True, "mu": 1, "eta": 1},
    ],
)
def test_adding_reference_constraint_to_existing_set_of_constraints(
    starting_constraints: Dict[str, Union[float, Literal["True"]]],
):
    cons = Constraints(starting_constraints)

    extra_constraints = {"alpha": 1, "beta": 2}

    if (len(cons.asdict) == 3) & (len(cons.reference) == 0):
        with pytest.raises(DiffcalcException):
            cons.asdict = {**cons.asdict, **extra_constraints}
        return

    cons.asdict = {**cons.asdict, **extra_constraints}

    assert pytest.approx(cons.beta) == 2
    assert cons.alpha is None


@pytest.mark.parametrize(
    ("starting_constraints"),
    [
        {"mu": 1},
        {"mu": 1, "eta": 1},
        {"delta": 1, "alpha": 1},
        {"delta": 1, "mu": 1},
        {"alpha": 1, "mu": 1},
        {"delta": 1, "alpha": 1, "mu": 1},
        {"delta": 1, "mu": 1, "eta": 1},
        {"alpha": 1, "mu": 1, "eta": 1},
        {"mu": 1, "eta": 1, "chi": 1},
    ],
)
def test_adding_sample_constraint_to_existing_set_of_constraints(
    starting_constraints: Dict[str, Union[float, Literal["True"]]],
):
    cons = Constraints(starting_constraints)

    if (len(cons.asdict) == 3) & (len(cons.sample) > 1):
        with pytest.raises(DiffcalcException):
            cons.phi = 1

        return

    cons.asdict = {**cons.asdict, "phi": 1.0}

    assert cons.phi == 1.0


@pytest.mark.parametrize(
    ("con_dict", "expected_lines"),
    [
        ({}, ["!   3 more constraints required"]),
        ({"nu": 9.12343}, ["!   2 more constraints required", "    nu   : 9.1234"]),
        ({"nu": None}, ["!   3 more constraints required"]),
        ({"a_eq_b": True}, ["!   2 more constraints required", "    a_eq_b"]),
        (
            {"naz": 9.12343, "a_eq_b": True},
            ["!   1 more constraint required", "    naz  : 9.1234", "    a_eq_b"],
        ),
        (
            {"naz": 9.12343, "a_eq_b": True, "mu": 9.12343},
            ["    naz  : 9.1234", "    a_eq_b", "    mu   : 9.1234"],
        ),
    ],
)
def test_report_constraints_lines(
    cm: Constraints,
    con_dict: Dict[str, Union[float, Literal["True"]]],
    expected_lines: List[str],
):
    cm.asdict = con_dict
    lines = cm._report_constraints_lines()

    assert np.all(lines == expected_lines)


def _constrain(self, *args):
    for con in args:
        cm.constrain(con)


@pytest.mark.parametrize(
    ("con_dict"),
    [
        {"naz": 1, "alpha": 2, "mu": 3},
        {"qaz": 1, "alpha": 2, "mu": 3},
        {"beta": 1, "mu": 2, "chi": 3},
        {"beta": 0, "mu": 0, "chi": 90},
        {"beta": 0, "mu": 0, "phi": 90},
        {"beta": 0, "eta": 0, "chi": 90},
        {"beta": 0, "eta": 0, "phi": 90},
        {"beta": 0, "chi": 0, "phi": 90},
        {"qaz": 0, "mu": 0, "chi": 90},
        {"qaz": 0, "mu": 0, "eta": 90},
        {"qaz": 0, "mu": 0, "phi": 90},
        {"qaz": 0, "eta": 0, "chi": 90},
        {"qaz": 0, "eta": 0, "phi": 90},
        {"qaz": 0, "phi": 0, "chi": 90},
        {"qaz": 0, "mu": 0, "bisect": True},
        {"qaz": 0, "eta": 0, "bisect": True},
        {"qaz": 0, "omega": 0, "bisect": True},
        {"mu": 0, "eta": 0, "chi": 0},
        {"mu": 0, "eta": 0, "phi": 0},
        {"mu": 0, "chi": 0, "phi": 0},
        {"eta": 0, "chi": 0, "phi": 0},
    ],
)
def test_implemented_modes(con_dict: Dict[str, Union[float, Literal["True"]]]):
    cons = Constraints(con_dict)

    assert cons.is_current_mode_implemented() is True


@pytest.mark.parametrize(
    ("con_dict"),
    [
        {"naz": 1},
        {"eta": 0, "chi": 0, "bisect": True},
    ],
)
def test_non_implemented_modes(con_dict: Dict[str, Union[float, Literal["True"]]]):
    cons = Constraints(con_dict)

    if len(con_dict) < 3:
        with pytest.raises(ValueError):
            cons.is_current_mode_implemented()
    else:
        assert cons.is_current_mode_implemented() is False


def test_set_constraint_with_wrong_type_fails():
    cons = Constraints()

    try:
        cons.delta = True
    except DiffcalcException as e:
        assert (
            e.args[0]
            == f"Constraint delta requires numerical value. Found {bool} instead."
        )

    try:
        cons.a_eq_b = 1
    except DiffcalcException as e:
        assert (
            e.args[0]
            == f"Constraint a_eq_b requires boolean True value. Found 1 instead."
        )


def test_as_tuple_getter():
    cons = Constraints({"mu": 0, "a_eq_b": True})

    assert "a_eq_b" in cons.astuple
    assert ("mu", 0) in cons.astuple
    assert type(cons.astuple) is tuple


def test_setting_already_active_constraint():
    cons = Constraints({"psi": 0, "mu": 0})

    cons.mu = 90

    assert cons.mu == 90


def test_serialisation(cm):
    cm.asdict = {"alpha": 1, "mu": 2, "phi": 1, "beta": 2}
    cm_json = cm.asdict
    assert Constraints(cm_json).asdict == cm.asdict
