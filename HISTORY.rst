=======
History
=======

0.4.0 (2022-08-12)
------------------

* Add fromdict/asdict to support serialisation of HKLCalculation object. (`#52 <https://github.com/DiamondLightSource/diffcalc-core/pull/52>`_)
* Add support for Python 3.10 and drop support for Python 3.6 EOL. (`#48 <https://github.com/DiamondLightSource/diffcalc-core/pull/48>`_)
* Fix hkl calculation when reference vector is parallel to incident x-ray beam.
* Raise exception if azimuthal orientation of scattering plane is degenerate in
a geometry with a reference vector constraint.
* Find solutions for 180 degree backscattering geometry.

0.3.0 (2021-05-04)
------------------

* Add indegrees attribute in Position and Constraints class for angle units.
* Add asdegrees, asradians methods for converting Position and Constraints
attributes between degrees and radians.
* Add asdegrees parameter to get_position and get_virtal_angles methods for
returning calculated angles in degrees or radians.
* Add fields attribute to Position class with angle names.
* Remove TORAD, TODEG constants.
* Change the default angles units in Position class to degrees.
* Fix manually set UB matrix type compatibility issue.

0.2.1 (2021-04-20)
------------------

* Fix reported miscut parameters to exclude azimuthal component.

0.2.0 (2021-04-14)
------------------

* Add support for Python version >= 3.6.

0.1.1 (2021-04-06)
------------------

* Add versions for package dependencies.

0.1.0 (2021-04-06)
------------------

* First release on PyPI.
