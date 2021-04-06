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

from math import pi

from diffcalc.ub.calc import UBCalculation
from numpy import array

from tests.diffcalc.scenarios import PosFromI16sEuler
from tests.tools import matrixeq_

# newub 'cubic'                   <-->  reffile('cubic)
# setlat 'cubic' 1 1 1 90 90 90   <-->  latt([1,1,1,90,90,90])
# pos wl 1                        <-->  BLi.setWavelength(1)
#                                <-->  c2th([0,0,1]) --> 60
# pos sixc [0 60 0 30 1 1]        <-->  pos euler [1 1 30 0 60 0]
# addref 1 0 0                    <-->  saveref('100',[1, 0, 0])
# pos chi 91                      <-->
# addref 0 0 1                    <-->  saveref('100',[1, 0, 0]) ; showref()
#                                      ubm('100','001')
#                                      ubm() ->
# array('d', [0.9996954135095477, -0.01745240643728364, -0.017449748351250637,
# 0.01744974835125045, 0.9998476951563913, -0.0003045864904520898,
# 0.017452406437283505, -1.1135499981271473e-16, 0.9998476951563912])


UB1 = (
    array(
        (
            (0.9996954135095477, -0.01745240643728364, -0.017449748351250637),
            (0.01744974835125045, 0.9998476951563913, -0.0003045864904520898),
            (0.017452406437283505, -1.1135499981271473e-16, 0.9998476951563912),
        )
    )
    * (2 * pi)
)

EN1 = 12.39842
REF1a = PosFromI16sEuler(1, 1, 30, 0, 60, 0)
REF1b = PosFromI16sEuler(1, 91, 30, 0, 60, 0)


def testAgainstI16Results():
    ubcalc = UBCalculation("cubcalc")
    ubcalc.set_lattice("latt", 1, 1, 1, 90, 90, 90)
    ubcalc.add_reflection((1, 0, 0), REF1a, EN1, "100")
    ubcalc.add_reflection((0, 0, 1), REF1b, EN1, "001")
    ubcalc.calc_ub()
    matrixeq_(ubcalc.UB, UB1)
