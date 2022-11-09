"""Module providing diffractometer position definition and rotation matrices.

Diffractometer axes and rotation matrix definitions are following conventions
described in [1]_.

References
----------
.. [1] H. You. "Angle calculations for a '4S+2D' six-circle diffractometer"
       J. Appl. Cryst. (1999). 32, 614-623.
"""
from typing import Dict, Tuple, Union

import numpy as np
from diffcalc.util import I, ureg, x_rotation, y_rotation, z_rotation
from numpy.linalg import inv
from pint import Quantity, Unit


class Position:
    """Class representing diffractometer orientation.

    Diffractometer orientation corresponding to (4+2) geometry
    defined in H. You paper (add reference)

    Attributes
    ----------
    fields: Tuple[str, str, str, str, str, str]
        Tuple with angle names
    mu: Union[float, Quantity], default = 0.0
        mu angle value
    delta: Union[float, Quantity], default = 0.0
        delta angle value
    nu: Union[float, Quantity], default = 0.0
        nu angle value
    eta: Union[float, Quantity], default = 0.0
        eta angle value
    chi: Union[float, Quantity], default = 0.0
        chi angle value
    phi: Union[float, Quantity], default = 0.0
        phi angle value
    unit: Union[str, Unit], default = deg
        Angle units
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
        mu: Union[float, Quantity] = 0.0,
        delta: Union[float, Quantity] = 0.0,
        nu: Union[float, Quantity] = 0.0,
        eta: Union[float, Quantity] = 0.0,
        chi: Union[float, Quantity] = 0.0,
        phi: Union[float, Quantity] = 0.0,
        unit: Union[str, Unit] = ureg.degree,
    ):
        self.mu: Quantity = None
        self.delta: Quantity = None
        self.nu: Quantity = None
        self.eta: Quantity = None
        self.chi: Quantity = None
        self.phi: Quantity = None

        for name, val in zip(Position.fields, (mu, delta, nu, eta, chi, phi)):
            if isinstance(val, Quantity):
                setattr(self, name, val.to(unit))
            elif isinstance(unit, Unit):
                setattr(self, name, float(val) * unit)
            else:
                setattr(self, name, float(val) * ureg(unit))

    def __str__(self):
        """Represent Position object information as a string.

        Returns
        -------
        str
            Position object string representation.
        """
        return (
            f"Position({', '.join((f'{k}: {v:.4f}' for k, v in self.asdict.items()))})"
        )

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

    @property
    def asdegrees(self) -> Dict[str, float]:
        """Return dictionary of diffractometer angles in degrees.

        Returns
        -------
        Dict[str, float]
            Dictionary of axis names object with angles in degrees.
        """
        pos_in_deg = {k: v.to("degree").magnitude for k, v in self.asdict.items()}
        return pos_in_deg

    @property
    def asradians(self) -> Dict[str, float]:
        """Return dictionary of diffractometer angles in radians.

        Returns
        -------
        Dict[str, float]
            Dictionary of axis names with angles in radians.
        """
        pos_in_rad = {k: v.to("radian").magnitude for k, v in self.asdict.items()}
        return pos_in_rad

    @property
    def asdict(self) -> Dict[str, Quantity]:
        """Return dictionary of diffractometer angle quantities.

        Returns
        -------
        Dict[str, Quantity]
            Dictionary of axis names and angle quantities.
        """
        return {field: getattr(self, field) for field in self.fields}

    @property
    def astuple(
        self,
    ) -> Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity]:
        """Return tuple of diffractometer angle quantities.

        Returns
        -------
        Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity]
            Tuple of diffractometer angle quantities.
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


# TODO: these below functions are all variations, you can shorten it.


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
