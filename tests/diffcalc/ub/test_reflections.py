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

from diffcalc.hkl.geometry import Position
from diffcalc.ub.reference import ReflectionList


class TestReflectionList:
    def setup_method(self):
        self.reflist = ReflectionList()
        pos = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        self.reflist.add_reflection((1, 2, 3), pos, 1000, "ref1")
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66)
        self.reflist.add_reflection((1.1, 2.2, 3.3), pos, 1100, "ref2")

    def test_str(self):
        res = str(self.reflist)
        assert (
            res
            == """     ENERGY     H     K     L        MU    DELTA       NU      ETA      CHI      PHI  TAG
   1 1000.000  1.00  2.00  3.00    0.1000   0.2000   0.3000   0.4000   0.5000   0.6000  ref1
   2 1100.000  1.10  2.20  3.30    0.1100   0.2200   0.3300   0.4400   0.5500   0.6600  ref2"""
        )

    def test_add_reflection(self):
        assert len(self.reflist) == 2
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66)
        self.reflist.add_reflection((11.1, 12.2, 13.3), pos, 1100, "ref2")

    def test_get_reflection(self):
        answered = self.reflist.get_reflection(1).astuple
        pos = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        desired = ((1, 2, 3), pos.astuple, 1000, "ref1")
        assert answered == desired
        answered = self.reflist.get_reflection("ref1").astuple
        assert answered == desired

    def test_remove_reflection(self):
        self.reflist.remove_reflection(1)
        answered = self.reflist.get_reflection(1).astuple
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66)
        desired = ((1.1, 2.2, 3.3), pos.astuple, 1100, "ref2")
        assert answered == desired
        self.reflist.remove_reflection("ref2")
        assert self.reflist.reflections == []

    def test_edit_reflection(self):
        ps = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        self.reflist.edit_reflection(1, (10, 20, 30), ps, 1000, "new1")
        assert self.reflist.get_reflection(1).astuple == (
            (10, 20, 30),
            ps.astuple,
            1000,
            "new1",
        )
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66)
        assert self.reflist.get_reflection(2).astuple == (
            (1.1, 2.2, 3.3),
            pos.astuple,
            1100,
            "ref2",
        )
        self.reflist.edit_reflection("ref2", (1.1, 2.2, 3.3), pos, 1100, "new2")
        assert self.reflist.get_reflection("new2").astuple == (
            (1.1, 2.2, 3.3),
            pos.astuple,
            1100,
            "new2",
        )
        self.reflist.edit_reflection("new2", (10, 20, 30), pos, 1100, "new1")
        assert self.reflist.get_reflection("new1").astuple == (
            (10, 20, 30),
            ps.astuple,
            1000,
            "new1",
        )

    def test_swap_reflection(self):
        self.reflist.swap_reflections(1, 2)
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66)
        assert self.reflist.get_reflection(1).astuple == (
            (1.1, 2.2, 3.3),
            pos.astuple,
            1100,
            "ref2",
        )
        pos = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        assert self.reflist.get_reflection(2).astuple == (
            (1, 2, 3),
            pos.astuple,
            1000,
            "ref1",
        )
        self.reflist.swap_reflections("ref1", "ref2")
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66)
        assert self.reflist.get_reflection(2).astuple == (
            (1.1, 2.2, 3.3),
            pos.astuple,
            1100,
            "ref2",
        )
        pos = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        assert self.reflist.get_reflection(1).astuple == (
            (1, 2, 3),
            pos.astuple,
            1000,
            "ref1",
        )
