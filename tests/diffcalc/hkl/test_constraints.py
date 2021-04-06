###
# Copyright 2008-2011 Diamond Light Source Ltd.
# This file is part of Diffcalc.
#
# Diffcalc is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Diffcalc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Diffcalc.  If not, see <http://www.gnu.org/licenses/>.
###


import pytest
from diffcalc.hkl.constraints import Constraints, _con_type, _Constraint
from diffcalc.util import DiffcalcException

from tests.test_tools import eq_
from tests.tools import assert_dict_almost_equal


def joined(d1, d2):
    d1.update(d2)
    return d1


@pytest.fixture
def cm():
    return Constraints()


def test_init(cm):
    eq_(cm.asdict, dict())
    eq_(cm._detector, dict())
    eq_(cm._reference, dict())
    eq_(cm._sample, dict())


def test_dict_init():
    cm = Constraints({"nu": 0, "mu": 0, "a_eq_b": True})
    eq_(cm.asdict, {"nu": 0, "mu": 0, "a_eq_b": True})
    eq_(cm.astuple, (("nu", 0), "a_eq_b", ("mu", 0)))
    assert cm.nu == 0
    assert cm.mu == 0
    assert cm.a_eq_b is True
    assert cm.naz is None

    cm = Constraints({"nu": 0, "mu": 0, "a_eq_b": False})
    eq_(cm.asdict, {"nu": 0, "mu": 0})
    eq_(cm.astuple, (("nu", 0), ("mu", 0)))
    eq_(cm.nu, 0)
    eq_(cm.mu, 0)
    eq_(cm.a_eq_b, None)
    eq_(cm.naz, None)

    cm.asdict = {"nu": 0, "mu": 0, "a_eq_b": None}
    eq_(cm.asdict, {"nu": 0, "mu": 0})
    eq_(cm.astuple, (("nu", 0), ("mu", 0)))
    eq_(cm.nu, 0)
    eq_(cm.mu, 0)
    eq_(cm.a_eq_b, None)
    eq_(cm.naz, None)

    cm.asdict = {"nu": 3, "mu": 4, "a_eq_b": True}
    assert_dict_almost_equal(cm.asdict, {"nu": 3, "mu": 4, "a_eq_b": True})
    eq_(cm.astuple, (("nu", pytest.approx(3)), "a_eq_b", ("mu", pytest.approx(4))))
    assert cm.nu == pytest.approx(3)
    assert cm.mu == pytest.approx(4)
    eq_(cm.a_eq_b, True)
    eq_(cm.naz, None)

    cm = Constraints((("nu", 0), ("mu", 0), "a_eq_b"))
    eq_(cm.asdict, {"nu": 0, "mu": 0, "a_eq_b": True})
    eq_(cm.astuple, (("nu", 0), "a_eq_b", ("mu", 0)))

    cm = Constraints([("nu", 0), ("mu", 0), "a_eq_b"])
    eq_(cm.asdict, {"nu": 0, "mu": 0, "a_eq_b": True})
    eq_(cm.astuple, (("nu", 0), "a_eq_b", ("mu", 0)))

    cm = Constraints({("nu", 0), ("mu", 0), "a_eq_b"})
    eq_(cm.asdict, {"nu": 0, "mu": 0, "a_eq_b": True})
    eq_(cm.astuple, (("nu", 0), "a_eq_b", ("mu", 0)))


def test_set_init():
    cm = Constraints({("nu", 0), ("mu", 0), "a_eq_b"})
    eq_(cm.asdict, {"nu": 0, "mu": 0, "a_eq_b": True})
    eq_(cm.astuple, (("nu", 0), "a_eq_b", ("mu", 0)))
    eq_(cm.nu, 0)
    eq_(cm.mu, 0)
    eq_(cm.a_eq_b, True)
    eq_(cm.naz, None)

    cm = Constraints({("nu", 0), ("mu", 0)})
    eq_(cm.asdict, {"nu": 0, "mu": 0})
    eq_(cm.astuple, (("nu", 0), ("mu", 0)))
    eq_(cm.nu, 0)
    eq_(cm.mu, 0)
    eq_(cm.a_eq_b, None)
    eq_(cm.naz, None)

    cm.astuple = (("nu", 3), "a_eq_b", ("mu", 4))
    assert_dict_almost_equal(cm.asdict, {"nu": 3, "mu": 4, "a_eq_b": True})

    eq_(cm.astuple, (("nu", pytest.approx(3)), "a_eq_b", ("mu", pytest.approx(4))))
    assert cm.nu == pytest.approx(3)
    assert cm.mu == pytest.approx(4)
    eq_(cm.a_eq_b, True)
    eq_(cm.naz, None)


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


def test_unconstrain_okay(cm):
    eq_(cm.asdict, dict())
    cm.delta = 1.0
    cm.mu = 2
    eq_(cm._constrained, (cm._delta, cm._mu))
    eq_(cm.asdict, {"delta": 1.0, "mu": 2})
    del cm.delta
    cm.mu = None
    eq_(cm.asdict, dict())
    assert cm.delta is None


@pytest.mark.parametrize(
    "con_name",
    [
        "delta",
        "nu",
        "naz",
        "qaz",
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
def test_every_constraint(con_name, cm):
    assert getattr(cm, con_name) is None
    con = getattr(cm, "_" + con_name)
    assert type(con) is _Constraint
    if con._type is _con_type.VALUE:
        setattr(cm, con_name, 1.0)
        assert getattr(cm, con_name) == 1.0
        eq_(cm.asdict, {con_name: 1.0})
    elif con._type is _con_type.VOID:
        setattr(cm, con_name, True)
        assert getattr(cm, con_name) is True
        eq_(cm.asdict, {con_name: True})
    else:
        raise TypeError("Invalid constraint type setting.")
    delattr(cm, con_name)
    assert getattr(cm, con_name) is None
    eq_(cm.asdict, {})
    eq_(cm.asdict, dict())


def test_clear_constraints(cm):
    cm.delta = 1
    cm.mu = 2
    cm.clear()
    eq_(cm.asdict, dict())
    assert cm.delta is None
    assert cm.mu is None


def test_constrain_det(cm, pre={}):
    assert_dict_almost_equal(cm.asdict, pre)
    cm.delta = 1
    assert_dict_almost_equal(cm.asdict, joined({"delta": 1}, pre))
    cm.naz = 1
    assert_dict_almost_equal(cm.asdict, joined({"naz": 1}, pre))
    cm.delta = 1
    assert_dict_almost_equal(cm.asdict, joined({"delta": 1}, pre))


def test_constrain_det_one_preexisting_ref(cm):
    cm.alpha = 2.0
    test_constrain_det(cm, {"alpha": 2.0})


def test_constrain_det_one_preexisting_samp(cm):
    cm.phi = 3.0
    test_constrain_det(cm, {"phi": 3.0})


def test_constrain_det_one_preexisting_samp_and_ref(cm):
    cm.alpha = 2.1
    cm.phi = 3.2
    test_constrain_det(cm, {"alpha": 2.1, "phi": 3.2})


def test_constrain_det_two_preexisting_samp(cm):
    cm.chi = 4.3
    cm.phi = 5.6
    test_constrain_det(cm, {"chi": 4.3, "phi": 5.6})


def test_constrain_det_three_preexisting_other(cm):
    cm.alpha = 1
    cm.phi = 2
    cm.chi = 3
    try:
        cm.delta = 4
        assert False
    except DiffcalcException as e:
        eq_(
            e.args[0],
            (
                "Cannot set delta constraint. First un-constrain one of the"
                "\nangles alpha, chi, phi."
            ),
        )


def test_constrain_det_three_preexisting_samp(cm):
    cm.phi = 1
    cm.chi = 2
    cm.eta = 3
    try:
        cm.delta = 4
        assert False
    except DiffcalcException as e:
        eq_(
            e.args[0],
            "Cannot set delta constraint. First un-constrain one of the"
            "\nangles chi, eta, phi.",
        )


def test_constrain_ref(cm, pre={}):
    assert_dict_almost_equal(cm.asdict, pre)
    cm.alpha = 1.0
    assert_dict_almost_equal(cm.asdict, joined({"alpha": 1.0}, pre))
    cm.alpha = 1.0
    assert_dict_almost_equal(cm.asdict, joined({"alpha": 1.0}, pre))
    cm.beta = 1.0
    assert_dict_almost_equal(cm.asdict, joined({"beta": 1.0}, pre))


def test_constrain_ref_one_preexisting_det(cm):
    cm.delta = 2
    test_constrain_ref(cm, {"delta": 2})


def test_constrain_ref_one_preexisting_samp(cm):
    cm.phi = 3
    test_constrain_ref(cm, {"phi": 3})


def test_constrain_ref_one_preexisting_samp_and_det(cm):
    cm.delta = 1
    cm.phi = 2
    test_constrain_ref(cm, {"delta": 1, "phi": 2})


def test_constrain_ref_two_preexisting_samp(cm):
    cm.chi = 1
    cm.phi = 2
    test_constrain_ref(cm, {"chi": 1, "phi": 2})


def test_constrain_ref_three_preexisting_other(cm):
    cm.delta = 1
    cm.phi = 2
    cm.chi = 3
    try:
        cm.alpha = 1
        assert False
    except DiffcalcException as e:
        eq_(
            e.args[0],
            "Cannot set alpha constraint. First un-constrain one of the"
            "\nangles chi, delta, phi.",
        )


def test_constrain_ref_three_preexisting_samp(cm):
    cm.phi = 1
    cm.chi = 2
    cm.eta = 3
    try:
        cm.delta = 1
        assert False
    except DiffcalcException as e:
        eq_(
            e.args[0],
            "Cannot set delta constraint. First un-constrain one of the"
            "\nangles chi, eta, phi.",
        )


def test_constrain_samp_when_one_free(cm, pre={}):
    assert_dict_almost_equal(cm.asdict, pre)
    cm.phi = 1.0
    assert_dict_almost_equal(cm.asdict, joined({"phi": 1.0}, pre))
    cm.phi = 1.0
    assert_dict_almost_equal(cm.asdict, joined({"phi": 1.0}, pre))


def test_constrain_samp_one_preexisting_samp(cm):
    cm.chi = 2.0
    test_constrain_samp_when_one_free(cm, {"chi": 2.0})


def test_constrain_samp_two_preexisting_samp(cm):
    cm.chi = 1
    cm.eta = 2
    test_constrain_samp_when_one_free(cm, {"chi": 1, "eta": 2})


def test_constrain_samp_two_preexisting_other(cm):
    cm.delta = 1
    cm.alpha = 2
    test_constrain_samp_when_one_free(cm, {"delta": 1, "alpha": 2})


def test_constrain_samp_two_preexisting_one_det(cm):
    cm.delta = 1
    cm.eta = 1
    test_constrain_samp_when_one_free(cm, {"delta": 1, "eta": 1})


def test_constrain_samp_two_preexisting_one_ref(cm):
    cm.alpha = 3
    cm.eta = 2
    test_constrain_samp_when_one_free(cm, {"alpha": 3, "eta": 2})


def test_constrain_samp_three_preexisting_only_one_samp(cm):
    cm.delta = 3
    cm.alpha = 4
    cm.eta = 5
    cm.phi = 1
    assert_dict_almost_equal(cm.asdict, {"delta": 3, "alpha": 4, "phi": 1})
    cm.phi = 2
    assert_dict_almost_equal(cm.asdict, {"delta": 3, "alpha": 4, "phi": 2})


def test_constrain_samp_three_preexisting_two_samp_one_det(cm):
    cm.delta = 1
    cm.eta = 2
    cm.chi = 3
    try:
        cm.phi = 4
        assert False
    except DiffcalcException as e:
        eq_(
            e.args[0],
            "Cannot set phi constraint. First un-constrain one of the"
            "\nangles chi, delta, eta.",
        )


def test_constrain_samp_three_preexisting_two_samp_one_ref(cm):
    cm.alpha = 2
    cm.eta = 3
    cm.chi = 4
    try:
        cm.phi = 4
        assert False
    except DiffcalcException as e:
        eq_(
            e.args[0],
            "Cannot set phi constraint. First un-constrain one of the"
            "\nangles alpha, chi, eta.",
        )


def test_constrain_samp_three_preexisting_samp(cm):
    cm.mu = 1
    cm.eta = 2
    cm.chi = 3
    try:
        cm.phi = 4
        assert False
    except DiffcalcException as e:
        eq_(
            e.args[0],
            "Cannot set phi constraint. First un-constrain one of the"
            "\nangles chi, eta, mu.",
        )


def test_report_constraints_none(cm):
    eq_(cm._report_constraints_lines(), ["!   3 more constraints required"])


def test_report_constraints_one_with_value(cm):
    cm.nu = 9.12343
    eq_(
        cm._report_constraints_lines(),
        ["!   2 more constraints required", "    nu   : 9.1234"],
    )


def test_report_constraints_one_with_novalue(cm):
    cm.nu = None
    eq_(
        cm._report_constraints_lines(),
        ["!   3 more constraints required"],
    )


def test_report_constraints_one_with_valueless(cm):
    cm.a_eq_b = True
    eq_(
        set(cm._report_constraints_lines()),
        {"!   2 more constraints required", "    a_eq_b"},
    )


def test_report_constraints_one_with_two(cm):
    cm.naz = 9.12343
    cm.a_eq_b = True
    eq_(
        set(cm._report_constraints_lines()),
        {"!   1 more constraint required", "    naz  : 9.1234", "    a_eq_b"},
    )


def test_report_constraints_one_with_three(cm):
    cm.naz = 9.12343
    cm.a_eq_b = True
    cm.mu = 9.12343

    eq_(
        set(cm._report_constraints_lines()),
        {"    naz  : 9.1234", "    a_eq_b", "    mu   : 9.1234"},
    )


def _constrain(self, *args):
    for con in args:
        cm.constrain(con)


@pytest.mark.xfail(raises=ValueError)
def test_is_implemented_invalid(cm):
    cm.naz = 1
    cm.is_current_mode_implemented()


# 1 samp


def test_is_implemented_1_samp_naz(cm):
    cm.naz = 1
    cm.alpha = 2
    cm.mu = 3
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_1_samp_det(cm):
    cm.qaz = 1
    cm.alpha = 2
    cm.mu = 3
    eq_(cm.is_current_mode_implemented(), True)


# 2 samp + ref


def test_is_implemented_2_samp_ref_mu_chi(cm):
    cm.beta = 1
    cm.mu = 2
    cm.chi = 3
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_ref_mu90_chi0(cm):
    cm.beta = 0
    cm.mu = 0
    cm.chi = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_ref_mu_eta(cm):
    cm.beta = 0
    cm.mu = 0
    cm.eta = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_ref_mu_phi(cm):
    cm.beta = 0
    cm.mu = 0
    cm.phi = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_ref_eta_chi(cm):
    cm.beta = 0
    cm.eta = 0
    cm.chi = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_ref_eta_phi(cm):
    cm.beta = 0
    cm.eta = 0
    cm.phi = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_ref_chi_phi(cm):
    cm.beta = 0
    cm.phi = 0
    cm.chi = 90
    eq_(cm.is_current_mode_implemented(), True)


# 2 samp + det


def test_is_implemented_2_samp_det_mu_chi(cm):
    cm.qaz = 0
    cm.mu = 0
    cm.chi = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_det_mu_eta(cm):
    cm.qaz = 0
    cm.mu = 0
    cm.eta = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_det_mu_phi(cm):
    cm.qaz = 0
    cm.mu = 0
    cm.phi = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_det_eta_chi(cm):
    cm.qaz = 0
    cm.eta = 0
    cm.chi = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_det_eta_phi(cm):
    cm.qaz = 0
    cm.eta = 0
    cm.phi = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_det_chi_phi(cm):
    cm.qaz = 0
    cm.phi = 0
    cm.chi = 90
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_det_bisect_mu(cm):
    cm.qaz = 0
    cm.mu = 0
    cm.bisect = True
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_det_bisect_eta(cm):
    cm.qaz = 0
    cm.eta = 0
    cm.bisect = True
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_det_bisect_omega(cm):
    cm.qaz = 0
    cm.omega = 0
    cm.bisect = True
    eq_(cm.is_current_mode_implemented(), True)


# 3 samp


def test_is_implemented_3_samp_no_mu(cm):
    cm.eta = 0
    cm.chi = 0
    cm.phi = 1
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_3_samp_no_eta(cm):
    cm.mu = 0
    cm.chi = 0
    cm.phi = 1
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_3_samp_no_chi(cm):
    cm.eta = 0
    cm.chi = 0
    cm.phi = 1
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_3_samp_no_phi(cm):
    cm.eta = 0
    cm.mu = 0
    cm.chi = 1
    eq_(cm.is_current_mode_implemented(), True)


def test_set_fails(cm):
    try:
        cm.delta = True
        assert False
    except DiffcalcException as e:
        eq_(
            e.args[0],
            f"Constraint delta requires numerical value. Found {bool} instead.",
        )

    try:
        cm.a_eq_b = 1
        assert False
    except DiffcalcException as e:
        eq_(
            e.args[0],
            f"Constraint a_eq_b requires boolean value. Found {int} instead.",
        )
