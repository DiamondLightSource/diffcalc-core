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
