"""Module providing diffractometer position definition and rotation matrices.

Diffractometer axes and rotation matrix definitions are following conventions
described in [1]_.

References
----------
.. [1] H. You. "Angle calculations for a '4S+2D' six-circle diffractometer"
       J. Appl. Cryst. (1999). 32, 614-623.
"""
from math import degrees, radians
from typing import Dict, Tuple, Union

import numpy as np
from diffcalc.util import DiffcalcException, I, x_rotation, y_rotation, z_rotation
from numpy.linalg import inv
from pint import Quantity, UnitRegistry

Angle = Union[float, Quantity]
ureg = UnitRegistry()


class Position:
    """Class representing diffractometer orientation.

    Diffractometer orientation corresponding to (4+2) geometry
    defined in H. You paper (add reference)

    Attributes
    ----------
    fields: Tuple[str, str, str, str, str, str]
        Tuple with angle names
    mu: float, default = 0.0
        mu angle value
    delta: float, default = 0.0
        delta angle value
    nu: float, default = 0.0
        nu angle value
    eta: float, default = 0.0
        eta angle value
    chi: float, default = 0.0
        chi angle value
    phi: float, default = 0.0
        phi angle value
    indegrees: bool, default = True
        If True, arguments are angles in degrees.
    """

    fields: Tuple[str, str, str, str, str, str] = (
        "mu",
        "delta",
        "nu",
        "eta",
        "chi",
        "phi",
    )

    def __init__(
        self,
        mu: Angle = 0.0,
        delta: Angle = 0.0,
        nu: Angle = 0.0,
        eta: Angle = 0.0,
        chi: Angle = 0.0,
        phi: Angle = 0.0,
    ):
        self.mu: Angle = mu
        self.delta: Angle = delta
        self.nu: Angle = nu
        self.eta: Angle = eta
        self.chi: Angle = chi
        self.phi: Angle = phi

        quantities_dimensionless = [
            getattr(self, field).dimensionless
            for field in self.fields
            if isinstance(getattr(self, field), Quantity)
        ]

        if False in quantities_dimensionless:
            raise DiffcalcException(
                "found non dimensionless field for Position object. If using pint to "
                + "define quantities, use either .deg or .rad on unit registry."
            )

    def __str__(self):
        """Represent Position object information as a string.

        Returns
        -------
        str
            Position object string representation.
        """
        values = {
            key: value.magnitude if isinstance(value, Quantity) else value
            for key, value in self.asdict.items()
        }
        return f"Position({', '.join((f'{k}: {v:.4f}' for k, v in values.items()))})"

    def __eq__(self, other):
        """Check if two Position objects are equivalent.
        This compares their underlying angle values, which are stored in radians,
        rather than the "public" variables the user can set/get.
        """
        if isinstance(other, Position):
            return (
                (self.mu == other.mu)
                & (self.delta == other.delta)
                & (self.nu == other.nu)
                & (self.eta == other.eta)
                & (self.chi == other.chi)
                & (self.phi == other.phi)
            )

        return False

    def __eq__(self, other):
        """Check if two Position objects are equivalent.

        This compares their underlying angle values, which are stored in radians,
        rather than the "public" variables the user can set/get.
        """
        if isinstance(other, Position):
            return (
                (self._mu == other._mu)
                & (self._delta == other._delta)
                & (self._nu == other._nu)
                & (self._eta == other._eta)
                & (self._chi == other._chi)
                & (self._phi == other._phi)
            )

        return False

    @classmethod
    def asdegrees(cls, pos: "Position") -> "Position":
        """Create new Position object with angles in degrees.

        Parameters
        ----------
        pos: Position
            Input Position object

        Returns
        -------
        Position
            New Position object with angles in degrees.
        """
        pos_in_deg = {k: degrees(v) * ureg.deg for k, v in pos.asdict.items()}
        return cls(**pos_in_deg)

    @classmethod
    def asradians(cls, pos: "Position") -> "Position":
        """Create new Position object with angles in radians.

        Parameters
        ----------
        pos: Position
            Input Position object

        Returns
        -------
        Position
            New Position object with angles in radians.
        """
        pos_in_rad = {k: radians(v) for k, v in pos.asdict.items()}
        return cls(**pos_in_rad)

    @property
    def asdict(self) -> Dict[str, float]:
        """Return dictionary of diffractometer angles.

        Returns
        -------
        Dict[str, float]
            Dictionary of axis names and angle values.
        """
        return {field: getattr(self, field) for field in self.fields}

    @property
    def astuple(self) -> Tuple[float, float, float, float, float, float]:
        """Return tuple of diffractometer angles.

        Returns
        -------
        Tuple[float, float, float, float, float, float]
            Tuple of angle values.
        """
        mu, delta, nu, eta, chi, phi = tuple(
            getattr(self, field) for field in self.fields
        )
        return mu, delta, nu, eta, chi, phi


def get_rotation_matrices(
    pos: Position,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
    """Create rotation matrices corresponding to the diffractometer axes.

    Parameters
    ----------
    pos: Position
        Position object containing set of diffractometer angles

    Returns
    -------
    Tuple[np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray]
        Tuple containing set of rotation matrices corresponding to
        input diffractometer angle values.
    """
    mu, delta, nu, eta, chi, phi = pos.astuple
    return (
        rot_MU(mu),
        rot_DELTA(delta),
        rot_NU(nu),
        rot_ETA(eta),
        rot_CHI(chi),
        rot_PHI(phi),
    )


def rot_NU(nu: float) -> np.ndarray:
    """Return rotation matrix corresponding to nu axis.

    Parameters
    ----------
        nu: float
        nu axis angle

    Returns
    -------
    np.ndarray
        Rotation matrix as a NumPy array.
    """
    return x_rotation(nu)


def rot_DELTA(delta: float) -> np.ndarray:
    """Return rotation matrix corresponding to delta axis.

    Parameters
    ----------
        delta: float
        delta axis angle

    Returns
    -------
    np.ndarray
        Rotation matrix as a NumPy array.
    """
    return z_rotation(-delta)


def rot_MU(mu_or_alpha: float) -> np.ndarray:
    """Return rotation matrix corresponding to mu axis.

    Parameters
    ----------
        mu: float
        mu axis angle

    Returns
    -------
    np.ndarray
        Rotation matrix as a NumPy array.
    """
    return x_rotation(mu_or_alpha)


def rot_ETA(eta: float) -> np.ndarray:
    """Return rotation matrix corresponding to eta axis.

    Parameters
    ----------
        eta: float
        eta axis angle

    Returns
    -------
    np.ndarray
        Rotation matrix as a NumPy array.
    """
    return z_rotation(-eta)


def rot_CHI(chi: float) -> np.ndarray:
    """Return rotation matrix corresponding to chi axis.

    Parameters
    ----------
        chi: float
        chi axis angle

    Returns
    -------
    np.ndarray
        Rotation matrix as a NumPy array.
    """
    return y_rotation(chi)


def rot_PHI(phi: float) -> np.ndarray:
    """Return rotation matrix corresponding to phi axis.

    Parameters
    ----------
        phi: float
        phi axis angle

    Returns
    -------
    np.ndarray
        Rotation matrix as a NumPy array.
    """
    return z_rotation(-phi)


def get_q_phi(pos: Position) -> np.ndarray:
    """Calculate scattering vector in laboratory frame.

    Calculate hkl in the phi frame in units of 2 * pi / lambda.

    Parameters
    ----------
    pos: object
        Diffractometer angles in radians.

    Returns
    -------
    matrix:
        Scattering vector coordinates corresponding to the input position.
    """
    [MU, DELTA, NU, ETA, CHI, PHI] = get_rotation_matrices(pos)
    # Equation 12: Compute the momentum transfer vector in the lab  frame
    y = np.array([[0], [1], [0]])
    q_lab = (NU @ DELTA - I) @ y
    # Transform this into the phi frame.
    return np.array(inv(PHI) @ inv(CHI) @ inv(ETA) @ inv(MU) @ q_lab)
