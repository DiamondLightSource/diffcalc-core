"""Top-level package for diffcalc-core."""

from pint import UnitRegistry, set_application_registry

ureg = UnitRegistry()
Q = ureg.Quantity
set_application_registry(ureg)

__author__ = """Diamond Light Source Ltd. - Scientific Software"""
__email__ = "scientificsoftware@diamond.ac.uk"
__version__ = "0.4.0"
