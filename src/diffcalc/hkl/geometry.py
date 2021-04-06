"""Module providing diffractometer position definition and rotation matrices.

Diffractometer axes and rotation matrix definitions are following conventions
described in [1]_.

References
----------
.. [1] H. You. "Angle calculations for a '4S+2D' six-circle diffractometer"
       J. Appl. Cryst. (1999). 32, 614-623.
"""
import dataclasses
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from diffcalc.util import I, x_rotation, y_rotation, z_rotation
from numpy.linalg import inv


@dataclass
class Position:
    """Class representing diffractometer orientation.

    Diffractometer orientation corresponding to (4+2) geometry
    defined in H. You paper (add reference)

    Attributes
    ----------
    mu: float, default =0
        mu angle value
    delta: float, default =0
        delta angle value
    nu: float, default =0
        nu angle value
    eta: float, default =0
        eta angle value
    chi: float, default =0
        chi angle value
    phi: float, default =0
        phi angle value
    """

    mu: float = 0
    delta: float = 0
    nu: float = 0
    eta: float = 0
    chi: float = 0
    phi: float = 0

    @property
    def asdict(self) -> Dict[str, float]:
        """Return dictionary of diffractometer angles.

        Returns
        -------
        Dict[str, float]
            Dictionary of axis names and angle values.
        """
        return dataclasses.asdict(self)

    @property
    def astuple(self) -> Tuple[float, float, float, float, float, float]:
        """Return tuple of diffractometer angles.

        Returns
        -------
        Tuple[float, float, float, float, float, float]
            Tuple of angle values.
        """
        mu, delta, nu, eta, chi, phi = dataclasses.astuple(self)
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
    return inv(PHI) @ inv(CHI) @ inv(ETA) @ inv(MU) @ q_lab
