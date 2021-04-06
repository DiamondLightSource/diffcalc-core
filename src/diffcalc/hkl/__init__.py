"""Diffractometer position calculations.

.. currentmodule:: diffcalc.hkl

Routines for calculating miller indices for a given diffractometer position and
all diffractometer positions matching miller indices. This implementation
follows calculations published in [1]_.

.. autosummary::
   :toctree: generated/

   calc - Calculation of miller indices an diffractometer position.
   constraints - Diffractometer constraint management object.
   geometry - Diffractometer position object and rotation matrices.


References
----------
.. [1] H. You. "Angle calculations for a '4S+2D' six-circle diffractometer"
       J. Appl. Cryst. (1999). 32, 614-623.
"""
