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
from diffcalc.hkl.constraints import Constraints, ConstraintTypes, boolean_constraints
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
    eq_(cm.detector, dict())
    eq_(cm.reference, dict())
    eq_(cm.sample, dict())


def test_dict_init():
    cm = Constraints({"nu": 0, "mu": 0, "a_eq_b": True})
    eq_(cm.asdict, {"nu": 0, "mu": 0, "a_eq_b": True})
    assert cm.nu == 0
    assert cm.mu == 0
    assert cm.a_eq_b is True
    assert cm.naz is None

    cm = Constraints({"nu": 0, "mu": 0, "a_eq_b": False})
    eq_(cm.asdict, {"nu": 0, "mu": 0})
    eq_(cm.nu, 0)
    eq_(cm.mu, 0)
    eq_(cm.a_eq_b, None)
    eq_(cm.naz, None)

    cm = Constraints({"nu": 0, "mu": 0, "a_eq_b": None})
    eq_(cm.asdict, {"nu": 0, "mu": 0})
    eq_(cm.nu, 0)
    eq_(cm.mu, 0)
    eq_(cm.a_eq_b, None)
    eq_(cm.naz, None)

    cm = Constraints({"nu": 3, "mu": 4, "a_eq_b": True})
    assert_dict_almost_equal(cm.asdict, {"nu": 3, "mu": 4, "a_eq_b": True})
    assert cm.nu == pytest.approx(3)
    assert cm.mu == pytest.approx(4)
    eq_(cm.a_eq_b, True)
    eq_(cm.naz, None)


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
    assert cm.asdict == {"delta": 1.0, "mu": 2}
    cm.unset("delta")
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
    boolean_constraint = con_name in [con.name.lower() for con in boolean_constraints]
    assert getattr(cm, con_name) is None
    cm.set(con_name, 1.0)

    if boolean_constraint:
        assert cm.asdict == {con_name: True}
        cm.set(con_name, False)
        assert cm.asdict == {}
    else:
        assert cm.asdict == {con_name: 1.0}

    cm.unset(con_name)
    assert cm.asdict == {}


def test_clear_constraints(cm):
    cm.delta = 1
    cm.mu = 2
    cm.clear()
    eq_(cm.asdict, dict())
    assert cm.delta is None
    assert cm.mu is None


def test_constrain_det(cm, pre={}):
    assert_dict_almost_equal(cm.asdict, pre)
    cm.set("delta", 1.0)
    assert_dict_almost_equal(cm.asdict, joined({"delta": 1}, pre))
    cm.set("naz", 1.0)
    assert_dict_almost_equal(cm.asdict, joined({"naz": 1}, pre))
    cm.set("delta", 1.0)
    assert_dict_almost_equal(cm.asdict, joined({"delta": 1}, pre))


def test_constrain_det_one_preexisting_ref(cm):
    cm.set("alpha", 2.0)
    test_constrain_det(cm, {"alpha": 2.0})


def test_constrain_det_one_preexisting_samp(cm):
    cm.set("phi", 3.0)
    test_constrain_det(cm, {"phi": 3.0})


def test_constrain_det_one_preexisting_samp_and_ref(cm):
    cm.set("alpha", 2.1)
    cm.set("phi", 3.2)
    test_constrain_det(cm, {"alpha": 2.1, "phi": 3.2})


def test_constrain_det_two_preexisting_samp(cm):
    cm.set("chi", 4.3)
    cm.set("phi", 5.6)
    test_constrain_det(cm, {"chi": 4.3, "phi": 5.6})


def test_constrain_det_three_preexisting_other(cm):
    cm.set("alpha", 1.0)
    cm.set("phi", 2.0)
    cm.set("chi", 3.0)

    with pytest.raises(DiffcalcException):
        cm.set("delta", 4.0)


def test_constrain_det_three_preexisting_samp(cm):
    cm.set("phi", 1.0)
    cm.set("chi", 2.0)
    cm.set("eta", 3.0)

    with pytest.raises(DiffcalcException):
        cm.set("delta", 4.0)


def test_constrain_ref(cm, pre={}):
    assert_dict_almost_equal(cm.asdict, pre)
    cm.set("alpha", 1.0)
    assert_dict_almost_equal(cm.asdict, joined({"alpha": 1.0}, pre))
    cm.set("alpha", 1.0)
    assert_dict_almost_equal(cm.asdict, joined({"alpha": 1.0}, pre))
    cm.set("beta", 1.0)
    assert_dict_almost_equal(cm.asdict, joined({"beta": 1.0}, pre))


def test_constrain_ref_one_preexisting_det(cm):
    cm.set("delta", 2.0)
    test_constrain_ref(cm, {"delta": 2})


def test_constrain_ref_one_preexisting_samp(cm):
    cm.set("phi", 3)
    test_constrain_ref(cm, {"phi": 3})


def test_constrain_ref_one_preexisting_samp_and_det(cm):
    cm.set("delta", 1)
    cm.set("phi", 2)
    test_constrain_ref(cm, {"delta": 1, "phi": 2})


def test_constrain_ref_two_preexisting_samp(cm):
    cm.set("chi", 1)
    cm.set("phi", 2)
    test_constrain_ref(cm, {"chi": 1, "phi": 2})


def test_constrain_ref_three_preexisting_other(cm):
    cm.set("delta", 1)
    cm.set("phi", 2)
    cm.set("chi", 3)

    with pytest.raises(DiffcalcException):
        cm.set("alpha", 4.0)


def test_constrain_ref_three_preexisting_samp(cm):
    cm.set("phi", 1)
    cm.set("chi", 2)
    cm.set("eta", 3)

    with pytest.raises(DiffcalcException):
        cm.set("delta", 1)


def test_constrain_samp_when_one_free(cm, pre={}):
    assert_dict_almost_equal(cm.asdict, pre)
    cm.set("phi", 1.0)
    assert_dict_almost_equal(cm.asdict, joined({"phi": 1.0}, pre))
    cm.set("phi", 1.0)
    assert_dict_almost_equal(cm.asdict, joined({"phi": 1.0}, pre))


def test_constrain_samp_one_preexisting_samp(cm):
    cm.set("chi", 2.0)
    test_constrain_samp_when_one_free(cm, {"chi": 2.0})


def test_constrain_samp_two_preexisting_samp(cm):
    cm.set("chi", 1)
    cm.set("eta", 2)
    test_constrain_samp_when_one_free(cm, {"chi": 1, "eta": 2})


def test_constrain_samp_two_preexisting_other(cm):
    cm.set("delta", 1)
    cm.set("alpha", 2)
    test_constrain_samp_when_one_free(cm, {"delta": 1, "alpha": 2})


def test_constrain_samp_two_preexisting_one_det(cm):
    cm.set("delta", 1)
    cm.set("eta", 1)
    test_constrain_samp_when_one_free(cm, {"delta": 1, "eta": 1})


def test_constrain_samp_two_preexisting_one_ref(cm):
    cm.set("alpha", 3)
    cm.set("eta", 2)
    test_constrain_samp_when_one_free(cm, {"alpha": 3, "eta": 2})


def test_constrain_samp_three_preexisting_only_one_samp(cm):
    cm.set("delta", 3)
    cm.set("alpha", 4)
    cm.set("eta", 5)
    cm.set("phi", 1)
    assert_dict_almost_equal(cm.asdict, {"delta": 3, "alpha": 4, "phi": 1})
    cm.set("phi", 2)
    assert_dict_almost_equal(cm.asdict, {"delta": 3, "alpha": 4, "phi": 2})


def test_constrain_samp_three_preexisting_two_samp_one_det(cm):
    cm.set("delta", 1)
    cm.set("eta", 2)
    cm.set("chi", 3)

    with pytest.raises(DiffcalcException):
        cm.set("phi", 4)


def test_constrain_samp_three_preexisting_two_samp_one_ref(cm):
    cm.set("mu", 2)
    cm.set("eta", 3)
    cm.set("alpha", 4)

    with pytest.raises(DiffcalcException):
        cm.set("phi", 4)


def test_constrain_samp_three_preexisting_samp(cm):
    cm.set("mu", 1)
    cm.set("eta", 2)
    cm.set("chi", 3)

    with pytest.raises(DiffcalcException):
        cm.set("phi", 4)


def test_report_constraints_none(cm):
    eq_(cm._report_constraints_lines(), ["!   3 more constraints required"])


def test_report_constraints_one_with_value(cm):
    cm.set("nu", 9.12343)
    eq_(
        cm._report_constraints_lines(),
        ["!   2 more constraints required", "    nu   : 9.1234"],
    )


def test_report_constraints_one_with_novalue(cm):
    cm.set("nu", None)
    eq_(
        cm._report_constraints_lines(),
        ["!   3 more constraints required"],
    )


def test_report_constraints_one_with_valueless(cm):
    cm.set("a_eq_b", True)
    eq_(
        set(cm._report_constraints_lines()),
        {"!   2 more constraints required", "    a_eq_b"},
    )


def test_report_constraints_one_with_two(cm):
    cm.set("naz", 9.12343)
    cm.set("a_eq_b", True)
    eq_(
        set(cm._report_constraints_lines()),
        {"!   1 more constraint required", "    naz  : 9.1234", "    a_eq_b"},
    )


def test_report_constraints_one_with_three(cm):
    cm.set("naz", 9.12343)
    cm.set("a_eq_b", True)
    cm.set("mu", 9.12343)

    eq_(
        set(cm._report_constraints_lines()),
        {"    naz  : 9.1234", "    a_eq_b", "    mu   : 9.1234"},
    )


def _constrain(self, *args):
    for con in args:
        cm.constrain(con)


# 1 samp


def test_is_implemented_1_samp_naz(cm):
    cm.set("naz", 1)
    cm.set("alpha", 2)
    cm.set("mu", 3)
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_1_samp_det(cm):
    cm.set("qaz", 1)
    cm.set("alpha", 2)
    cm.set("mu", 3)
    eq_(cm.is_current_mode_implemented(), True)


# 2 samp + ref


def test_is_implemented_2_samp_ref_mu_chi(cm):
    cm.set("beta", 1)
    cm.set("mu", 2)
    cm.set("chi", 3)
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_ref_mu90_chi0(cm):
    cm.set("beta", 0)
    cm.set("mu", 0)
    cm.set("chi", 3.14 / 2)
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_ref_mu_eta(cm):
    cm.set("beta", 0)
    cm.set("mu", 0)
    cm.set("eta", 3.14 / 2)
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_ref_mu_phi(cm):
    cm.set("beta", 0)
    cm.set("mu", 0)
    cm.set("phi", 3.14 / 2)
    eq_(cm.is_current_mode_implemented(), True)


def test_is_implemented_2_samp_ref_eta_chi(cm):
    cm.set("beta", 0)
    cm.set("eta", 0)
    cm.set("chi", 3.14 / 2)
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


def test_serialisation():
    cm = Constraints({"alpha": 1, "mu": 2, "phi": 1, "beta": 2})
    cm_json = cm.asdict
    assert Constraints(cm_json).asdict == cm.asdict
