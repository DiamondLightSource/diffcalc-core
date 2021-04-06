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

from tests.tools import degrees_equivalent


class TestUtils:
    def testDegreesEqual(self):
        tol = 0.001
        assert degrees_equivalent(1, 1, tol)
        assert degrees_equivalent(1, -359, tol)
        assert degrees_equivalent(359, -1, tol)
        assert not degrees_equivalent(1.1, 1, tol)
        assert not degrees_equivalent(1.1, -359, tol)
        assert not degrees_equivalent(359.1, -1, tol)


class TestPosition:
    def testCompare(self):
        # Test the compare method
        pos1 = Position(1, 2, 3, 4, 5, 6)
        pos2 = Position(1.1, 2.1, 3.1, 4.1, 5.1, 6.1)

        assert pos1 == pos1
        assert pos1 != pos2
