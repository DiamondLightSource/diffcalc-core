"""Common types used throughout diffcalc-core, required by diffcalc_API."""

from dataclasses import dataclass


@dataclass
class CrystalData:
    """diffcalc.ub.crystal.Crystal objects are serialised in this way"""

    name: str
    system: str
    a: float  # pylint: disable=invalid-name
    b: float  # pylint: disable=invalid-name
    c: float  # pylint: disable=invalid-name
    alpha: float
    beta: float
    gamma: float
