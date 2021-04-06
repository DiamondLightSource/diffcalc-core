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

from math import sqrt

import pytest
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import ReferenceVector, UBCalculation
from diffcalc.util import TODEG, TORAD, DiffcalcException
from numpy import array

from tests.diffcalc import scenarios
from tests.diffcalc.scenarios import Pos
from tests.test_tools import eq_
from tests.tools import assert_iterable_almost_equal, mneq_


class TestUBCalculation:
    def setup_method(self):
        self._refineub_matrix = array(
            [
                [0.70711, 0.70711, 0.00000],
                [-0.70711, 0.70711, 0.00000],
                [0.00000, 0.00000, 1.00000],
            ]
        )

    def test_str(self):
        ubcalc = UBCalculation("test_str")
        ubcalc.n_phi = (0, 0, 1)
        ubcalc.surf_nphi = (0, 0, 1)
        ubcalc.set_lattice("xtal", "Cubic", 1)
        ubcalc.add_reflection((0, 0, 1), Position(0, 60, 0, 30, 0, 0), 12.4, "ref1")
        ubcalc.add_orientation(
            (0, 1, 0), (0, 1, 0), Position(1, 0, 0, 0, 2, 0), "orient1"
        )
        ubcalc.set_miscut(None, 2.0 * TORAD)

        assert (
            str(ubcalc)
            == """UBCALC

   name:      test_str

REFERNCE

   n_hkl:     -0.03490   0.00000   0.99939
   n_phi:      0.00000   0.00000   1.00000 <- set

SURFACE NORMAL

   n_hkl:     -0.03490   0.00000   0.99939
   n_phi:      0.00000   0.00000   1.00000 <- set

CRYSTAL

   name:          xtal

   a, b, c:    1.00000   1.00000   1.00000
              90.00000  90.00000  90.00000  Cubic

   B matrix:   6.28319   0.00000   0.00000
               0.00000   6.28319   0.00000
               0.00000   0.00000   6.28319

UB MATRIX

   U matrix:   0.99939   0.00000   0.03490
               0.00000   1.00000   0.00000
              -0.03490   0.00000   0.99939

   miscut:
      angle:   2.00000
       axis:   0.00000   1.00000   0.00000

   UB matrix:  6.27936   0.00000   0.21928
               0.00000   6.28319   0.00000
              -0.21928   0.00000   6.27936

REFLECTIONS

     ENERGY     H     K     L        MU    DELTA       NU      ETA      CHI      PHI  TAG
   1 12.400  0.00  0.00  1.00    0.0000  60.0000   0.0000  30.0000   0.0000   0.0000  ref1

CRYSTAL ORIENTATIONS

         H     K     L       X     Y     Z        MU    DELTA       NU      ETA      CHI      PHI  TAG
   1  0.00  1.00  0.00   0.00  1.00  0.00    1.0000   0.0000   0.0000   0.0000   2.0000   0.0000  orient1"""
        )

    def test_set_lattice(self):
        ubcalc = UBCalculation("testing_set_lattice")
        with pytest.raises(TypeError):
            ubcalc.set_lattice(1)
        with pytest.raises(TypeError):
            ubcalc.set_lattice(1, 2)
        with pytest.raises(TypeError):
            ubcalc.set_lattice("HCl")
        ubcalc.set_lattice("NaCl", 1.1)
        eq_(("NaCl", 1.1, 1.1, 1.1, 90, 90, 90), ubcalc.crystal.get_lattice())
        ubcalc.set_lattice("NaCl", 1.1, 2.2)
        eq_(("NaCl", 1.1, 1.1, 2.2, 90, 90, 90), ubcalc.crystal.get_lattice())
        ubcalc.set_lattice("NaCl", 1.1, 2.2, 3.3)
        eq_(("NaCl", 1.1, 2.2, 3.3, 90, 90, 90), ubcalc.crystal.get_lattice())
        ubcalc.set_lattice("NaCl", 1.1, 2.2, 3.3, 91)
        eq_(("NaCl", 1.1, 2.2, 3.3, 90, 91, 90), ubcalc.crystal.get_lattice())
        with pytest.raises(TypeError):
            ubcalc.set_lattice(("NaCl", 1.1, 2.2, 3.3, 91, 92))
        ubcalc.set_lattice("NaCl", 1.1, 2.2, 3.3, 91, 92, 93)
        assert_iterable_almost_equal(
            ("NaCl", 1.1, 2.2, 3.3, 91, 92, 92.99999999999999),
            ubcalc.crystal.get_lattice(),
        )

    def test_add_reflection(self):
        # start new ubcalc
        ubcalc = UBCalculation("testing_add_reflection")
        reflist = ubcalc.reflist  # for convenience

        pos1 = Position(1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
        pos2 = Position(2.1, 2.2, 2.3, 2.4, 2.5, 2.6)
        pos3 = (3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 1000)
        pos4 = Position(2.1, 2.2, 2.3, 2.4, 2.5, 2.6)
        #
        energy = 1.10
        ubcalc.add_reflection((1.1, 1.2, 1.3), pos1, energy)
        result = reflist.get_reflection(1)
        eq_(result.astuple, ((1.1, 1.2, 1.3), pos1.astuple, 1.10, None))

        energy = 2.10
        ubcalc.add_reflection((2.1, 2.2, 2.3), pos2, energy, "atag")
        result = reflist.get_reflection(2)
        eq_(result.astuple, ((2.1, 2.2, 2.3), pos2.astuple, 2.10, "atag"))

        with pytest.raises(TypeError):
            ubcalc.add_reflection((3.1, 3.2, 3.3), pos3, 3.10)
        with pytest.raises(IndexError):
            reflist.get_reflection(3)

        ubcalc.add_reflection((4.1, 4.2, 4.3), pos4, 4.10, "tag2")
        result = reflist.get_reflection(3)
        eq_(result.astuple, ((4.1, 4.2, 4.3), pos4.astuple, 4.10, "tag2"))

    def test_edit_reflection(self):
        pos1 = Position(1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
        pos2 = Position(2.1, 2.2, 2.3, 2.4, 2.5, 2.6)
        ubcalc = UBCalculation("testing_editref")
        ubcalc.add_reflection((1, 2, 3), pos1, 10, "tag1")
        energy = 11
        ubcalc.edit_reflection(1, (1.1, 2, 3.1), pos2, energy, "tag1")

        reflist = ubcalc.reflist  # for convenience
        result = reflist.get_reflection(1)
        eq_(result.astuple, ((1.1, 2, 3.1), pos2.astuple, 11, "tag1"))

    def test_swap_reflections(self):
        ubcalc = UBCalculation("testing_swapref")
        pos = Position(1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
        ubcalc.add_reflection((1, 2, 3), pos, 10, "tag1")
        ubcalc.add_reflection((1, 2, 3), pos, 10, "tag2")
        ubcalc.add_reflection((1, 2, 3), pos, 10, "tag3")
        ubcalc.swap_reflections(1, 3)
        ubcalc.swap_reflections(1, 3)
        ubcalc.swap_reflections(3, 1)  # end flipped
        reflist = ubcalc.reflist
        tag1 = reflist.get_reflection(1).tag
        tag2 = reflist.get_reflection(2).tag
        tag3 = reflist.get_reflection(3).tag
        eq_(tag1, "tag3")
        eq_(tag2, "tag2")
        eq_(tag3, "tag1")
        ubcalc.swap_reflections(1, 2)
        tag1 = reflist.get_reflection(1).tag
        tag2 = reflist.get_reflection(2).tag
        eq_(tag1, "tag2")
        eq_(tag2, "tag3")

    def test_delref(self):
        ubcalc = UBCalculation("testing_swapref")
        pos = Position(1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
        ubcalc.add_reflection((1, 2, 3), pos, 10, "tag1")
        reflist = ubcalc.reflist
        reflist.get_reflection(1)
        ubcalc.del_reflection(1)
        with pytest.raises(IndexError):
            reflist.get_reflection(1)
        with pytest.raises(IndexError):
            ubcalc.del_reflection(1)

    def test_add_orientation(self):
        ubcalc = UBCalculation("testing_add_orientation")
        orientlist = ubcalc.orientlist  # for convenience

        hkl1 = (1.1, 1.2, 1.3)
        hkl2 = (2.1, 2.2, 2.3)
        orient1 = (1.4, 1.5, 1.6)
        orient2 = (2.4, 2.5, 2.6)
        #
        ubcalc.add_orientation(hkl1, orient1)
        result = orientlist.get_orientation(1)
        trans_orient1 = array([orient1])
        eq_((result.h, result.k, result.l), hkl1)
        mneq_(array([[result.x, result.y, result.z]]), trans_orient1)
        eq_(result.tag, None)

        with pytest.raises(TypeError):
            ubcalc.add_orientation(hkl2, orient2, "atag")

        ubcalc.add_orientation(hkl2, orient2, tag="atag")
        result = orientlist.get_orientation(2)
        trans_orient2 = array([orient2])
        eq_((result.h, result.k, result.l), hkl2)
        mneq_(array([[result.x, result.y, result.z]]), trans_orient2)
        eq_(result.tag, "atag")

    def test_edit_orientation(self):
        hkl1 = (1.1, 1.2, 1.3)
        hkl2 = (1.1, 1.2, 3.1)
        orient1 = (1.4, 1.5, 1.6)
        orient2 = (2.4, 1.5, 2.6)
        ubcalc = UBCalculation("testing_editorient")
        ubcalc.add_orientation(hkl1, orient1, tag="tag1")
        ubcalc.edit_orientation(1, (1.1, 1.2, 3.1), orient2, None, "newtag")

        orientlist = ubcalc.orientlist
        result = orientlist.get_orientation(1)
        trans_orient2 = array([orient2])
        eq_((result.h, result.k, result.l), hkl2)
        mneq_(array([[result.x, result.y, result.z]]), trans_orient2)
        eq_(result.tag, "newtag")

    def test_swap_orientations(self):
        ubcalc = UBCalculation("testing_swap_orientations")
        hkl = (1.1, 1.2, 1.3)
        orient = (1.4, 1.5, 1.6)
        ubcalc.add_orientation(hkl, orient, tag="tag1")
        ubcalc.add_orientation(hkl, orient, tag="tag2")
        ubcalc.add_orientation(hkl, orient, tag="tag3")
        ubcalc.swap_orientations(1, 3)
        ubcalc.swap_orientations(1, 3)
        ubcalc.swap_orientations(3, 1)  # end flipped
        orientlist = ubcalc.orientlist
        tag1 = orientlist.get_orientation(1).tag
        tag2 = orientlist.get_orientation(2).tag
        tag3 = orientlist.get_orientation(3).tag
        eq_(tag1, "tag3")
        eq_(tag2, "tag2")
        eq_(tag3, "tag1")
        ubcalc.swap_orientations(1, 2)
        tag1 = orientlist.get_orientation(1).tag
        tag2 = orientlist.get_orientation(2).tag
        eq_(tag1, "tag2")
        eq_(tag2, "tag3")

    def test_del_orientation(self):
        ubcalc = UBCalculation("testing_del_orientation")
        hkl = (1.1, 1.2, 1.3)
        pos = (1.4, 1.5, 1.6)
        ubcalc.add_orientation(hkl, pos, tag="tag1")
        orientlist = ubcalc.orientlist
        orientlist.get_orientation(1)
        ubcalc.del_orientation(1)
        with pytest.raises(IndexError):
            orientlist.get_orientation(1)
        with pytest.raises(IndexError):
            ubcalc.del_orientation(1)

    def test_setu(self):
        # just test calling this method
        # self.ub.setu([[1,2,3],[1,2,3],[1,2,3]])
        ubcalc = UBCalculation("test_setu")
        setu = ubcalc.set_u
        with pytest.raises(TypeError):
            setu(1, 2)
        with pytest.raises(TypeError):
            setu(1)
        with pytest.raises(TypeError):
            setu("a")
        with pytest.raises(TypeError):
            setu([1, 2, 3])
        with pytest.raises(TypeError):
            setu([[1, 2, 3], [1, 2, 3], [1, 2]])
        # diffCalcException expected if no lattice set yet
        setu([[1, 2, 3], [1, 2, 3], [1, 2, 3]])  # check no exceptions only
        setu(((1, 2, 3), (1, 2, 3), (1, 2, 3)))  # check no exceptions only

    def test_setub(self):
        # just test calling this method
        ubcalc = UBCalculation("test_setub")
        setub = ubcalc.set_ub
        with pytest.raises(TypeError):
            setub(1)
        with pytest.raises(TypeError):
            setub("a")
        with pytest.raises(TypeError):
            setub([1, 2, 3])
        with pytest.raises(TypeError):
            setub([[1, 2, 3], [1, 2, 3], [1, 2]])
        setub([[1, 2, 3], [1, 2, 3], [1, 2, 3]])  # check no exceptions only
        setub(((1, 2, 3), (1, 2, 3), (1, 2, 3)))  # check no exceptions only

    def test_calcub(self):
        # not enough reflections:
        ubcalc = UBCalculation("test_calcub")
        with pytest.raises(DiffcalcException):
            ubcalc.calc_ub()

        for s in scenarios.sessions():
            ubcalc = UBCalculation("test_calcub")
            ubcalc.set_lattice(s.name, *s.lattice)
            r = s.ref1
            ubcalc.add_reflection((r.h, r.k, r.l), r.pos, r.energy, r.tag)
            r = s.ref2
            ubcalc.add_reflection((r.h, r.k, r.l), r.pos, r.energy, r.tag)
            ubcalc.calc_ub(s.ref1.tag, s.ref2.tag)
            mneq_(
                ubcalc.UB,
                array(s.umatrix) @ array(s.bmatrix),
                4,
                note="wrong UB matrix after calculating U",
            )

    def test_orientub(self):
        ubcalc = UBCalculation("test_orientub")
        # not enough orientations:
        with pytest.raises(DiffcalcException):
            ubcalc.calc_ub()

        s = scenarios.sessions()[1]
        ubcalc.set_lattice(s.name, *s.lattice)
        r1 = s.ref1
        orient1 = array([[1], [0], [0]]).T.tolist()[0]
        tag1 = "or" + r1.tag
        ubcalc.add_orientation((r1.h, r1.k, r1.l), orient1, tag=tag1)
        r2 = s.ref2
        orient2 = array([[0], [-1], [0]]).T.tolist()[0]
        tag2 = "or" + r2.tag
        ubcalc.add_orientation((r2.h, r2.k, r2.l), orient2, tag=tag2)
        ubcalc.calc_ub()
        mneq_(
            ubcalc.UB,
            array(s.umatrix) @ array(s.bmatrix),
            4,
            note="wrong UB matrix after calculating U",
        )
        ubcalc.calc_ub(tag1, tag2)
        mneq_(
            ubcalc.UB,
            array(s.umatrix) * array(s.bmatrix),
            4,
            note="wrong UB matrix after calculating U",
        )
        ubcalc.add_reflection((r1.h, r1.k, r1.l), r1.pos, r1.energy, r1.tag)
        ubcalc.calc_ub(r1.tag, tag2)
        mneq_(
            ubcalc.UB,
            array(s.umatrix) @ array(s.bmatrix),
            4,
            note="wrong UB matrix after calculating U",
        )
        ubcalc.add_reflection((r2.h, r2.k, r2.l), r2.pos, r2.energy, r2.tag)
        ubcalc.calc_ub(tag1, r2.tag)
        mneq_(
            ubcalc.UB,
            array(s.umatrix) @ array(s.bmatrix),
            4,
            note="wrong UB matrix after calculating U",
        )

    def test_refine_ub(self):
        ubcalc = UBCalculation("testing_refine_ub")
        ubcalc.set_lattice("xtal", 1, 1, 1, 90, 90, 90)
        ubcalc.set_miscut(None, 0)
        ubcalc.refine_ub(
            (1, 1, 0), Pos(mu=0, delta=60, nu=0, eta=30, chi=0, phi=0), 1.0, True, True
        )
        eq_(("xtal", sqrt(2.0), sqrt(2.0), 1, 90, 90, 90), ubcalc.crystal.get_lattice())
        mneq_(
            ubcalc.U,
            self._refineub_matrix,
            4,
            note="wrong U matrix after refinement",
        )

    def test_fit_ub(self):
        for s in scenarios.sessions()[-3:]:
            ubcalc = UBCalculation("test_fit_ub_matrix")
            a, b, c, alpha, beta, gamma = s.lattice
            for r in s.reflist:
                ubcalc.add_reflection((r.h, r.k, r.l), r.pos, r.energy, r.tag)
            ubcalc.set_lattice(s.name, s.system, *s.lattice)
            ubcalc.calc_ub(s.ref1.tag, s.ref2.tag)

            init_latt = (
                1.06 * a,
                1.07 * b,
                0.94 * c,
                1.05 * alpha,
                1.06 * beta,
                0.95 * gamma,
            )
            ubcalc.set_lattice(s.name, s.system, *init_latt)
            ubcalc.set_miscut([0.2, 0.8, 0.1], 3.0, True)

            ubcalc.fit_ub([r.tag for r in s.reflist], True, True)
            assert ubcalc.crystal.system == s.system
            mneq_(
                array([ubcalc.crystal.get_lattice()[1:]]),
                array([s.lattice]),
                2,
                note="wrong lattice after fitting UB",
            )
            mneq_(
                ubcalc.U,
                array(s.umatrix),
                3,
                note="wrong U matrix after fitting UB",
            )

    def test_get_ttheta_from_hkl(self):
        ubcalc = UBCalculation("test_get_ttheta_from_hkl")
        ubcalc.set_lattice("cube", 1, 1, 1, 90, 90, 90)
        assert ubcalc.get_ttheta_from_hkl((0, 0, 1), 12.39842) == pytest.approx(
            60 * TORAD
        )

    def test_miscut(self):
        ubcalc = UBCalculation("testsetmiscut")
        ubcalc.reference = ReferenceVector((0, 0, 1), False)
        ubcalc.set_lattice("cube", 1, 1, 1, 90, 90, 90)
        beam_axis = array([[0], [1], [0]]).T.tolist()[0]
        beam_maxis = array([[0], [-1], [0]]).T.tolist()[0]
        ubcalc.set_miscut(beam_axis, 30 * TORAD)
        mneq_(
            ubcalc.n_hkl,
            array([[-0.5000000], [0.00000], [0.8660254]]),
        )
        ubcalc.set_miscut(beam_axis, 15 * TORAD, True)
        mneq_(
            ubcalc.n_hkl,
            array([[-0.7071068], [0.00000], [0.7071068]]),
        )
        ubcalc.set_miscut(beam_maxis, 45 * TORAD, True)
        mneq_(ubcalc.n_hkl, array([[0.0], [0.0], [1.0]]))

    @pytest.mark.parametrize(
        ("axis", "angle", "hkl"),
        [((0, 1, 0), 10, (0, 0, 1)), ((1, 0, 0), 30, (0, 1, 1))],
    )
    def test_get_miscut_from_hkl(self, axis, angle, hkl):
        ubcalc = UBCalculation("testing_calc_miscut")
        ubcalc.set_lattice("xtal", 1, 1, 1, 90, 90, 90)
        ubcalc.set_miscut(axis, angle * TORAD)
        hklcalc = HklCalculation(ubcalc, Constraints({"delta": 0, "psi": 0, "eta": 0}))
        pos, _ = hklcalc.get_position(*hkl, 1.0)[0]
        ubcalc.set_miscut(None, 0)
        miscut, miscut_axis = ubcalc.get_miscut_from_hkl(hkl, pos)
        assert miscut == pytest.approx(angle)
        mneq_(
            array([axis]),
            array([miscut_axis]),
            4,
            note="wrong calculation for miscut axis",
        )

    @pytest.mark.parametrize(("axis", "angle"), [((0, 1, 0), 10), ((1, 0, 0), 30)])
    def test_get_miscut(self, axis, angle):
        ubcalc = UBCalculation("testing_calc_miscut")
        ubcalc.set_lattice("xtal", 1, 1, 1, 90, 90, 90)
        ubcalc.set_miscut(axis, angle * TORAD)
        test_angle, test_axis = ubcalc.get_miscut()
        assert test_angle * TODEG == pytest.approx(angle)
        mneq_(
            array([test_axis]),
            array([axis]),
            4,
            note="wrong calculation for miscut axis",
        )
