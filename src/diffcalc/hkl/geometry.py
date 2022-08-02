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
from diffcalc.util import I, x_rotation, y_rotation, z_rotation
from numpy.linalg import inv


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
        mu: float = 0.0,
        delta: float = 0.0,
        nu: float = 0.0,
        eta: float = 0.0,
        chi: float = 0.0,
        phi: float = 0.0,
        indegrees: bool = True,
    ):
        self._mu: float = radians(mu) if indegrees else mu
        self._delta: float = radians(delta) if indegrees else delta
        self._nu: float = radians(nu) if indegrees else nu
        self._eta: float = radians(eta) if indegrees else eta
        self._chi: float = radians(chi) if indegrees else chi
        self._phi: float = radians(phi) if indegrees else phi
        self.indegrees: bool = indegrees

    def __str__(self):
        """Represent Position object information as a string.

        Returns
        -------
        str
            Position object string representation.
        """
        if self.indegrees:
            return f"Position({', '.join((f'{k}: {v:.4f}' for k, v in self.asdict.items()))})"
        return f"Position({', '.join((f'{k}: {degrees(v):.4f}' for k, v in self.asdict.items()))})"

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
        res = cls(**pos.asdict, indegrees=pos.indegrees)
        res.indegrees = True
        return res

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
        res = cls(**pos.asdict, indegrees=pos.indegrees)
        res.indegrees = False
        return res

    @property
    def mu(self) -> Union[float, None]:
        """Value of of mu angle."""
        if self.indegrees:
            return degrees(self._mu)
        else:
            return self._mu

    @mu.setter
    def mu(self, val):
        if self.indegrees:
            self._mu = radians(val)
        else:
            self._mu = val

    @mu.deleter
    def mu(self):
        self._mu = None

    @property
    def delta(self) -> Union[float, None]:
        """Value of of delta angle."""
        if self.indegrees:
            return degrees(self._delta)
        else:
            return self._delta

    @delta.setter
    def delta(self, val):
        if self.indegrees:
            self._delta = radians(val)
        else:
            self._delta = val

    @delta.deleter
    def delta(self):
        self._delta = None

    @property
    def nu(self) -> Union[float, None]:
        """Value of of nu angle."""
        if self.indegrees:
            return degrees(self._nu)
        else:
            return self._nu

    @nu.setter
    def nu(self, val):
        if self.indegrees:
            self._nu = radians(val)
        else:
            self._nu = val

    @nu.deleter
    def nu(self):
        self._nu = None

    @property
    def eta(self) -> Union[float, None]:
        """Value of of eta angle."""
        if self.indegrees:
            return degrees(self._eta)
        else:
            return self._eta

    @eta.setter
    def eta(self, val):
        if self.indegrees:
            self._eta = radians(val)
        else:
            self._eta = val

    @eta.deleter
    def eta(self):
        self._eta = None

    @property
    def chi(self) -> Union[float, None]:
        """Value of of chi angle."""
        if self.indegrees:
            return degrees(self._chi)
        else:
            return self._chi

    @chi.setter
    def chi(self, val):
        if self.indegrees:
            self._chi = radians(val)
        else:
            self._chi = val

    @chi.deleter
    def chi(self):
        self._chi = None

    @property
    def phi(self) -> Union[float, None]:
        """Value of of phi angle."""
        if self.indegrees:
            return degrees(self._phi)
        else:
            return self._phi

    @phi.setter
    def phi(self, val):
        if self.indegrees:
            self._phi = radians(val)
        else:
            self._phi = val

    @phi.deleter
    def phi(self):
        self._phi = None

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
    pos_in_rad = Position.asradians(pos)
    mu, delta, nu, eta, chi, phi = pos_in_rad.astuple
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
    pos_in_rad = Position.asradians(pos)
    [MU, DELTA, NU, ETA, CHI, PHI] = get_rotation_matrices(pos_in_rad)
    # Equation 12: Compute the momentum transfer vector in the lab  frame
    y = np.array([[0], [1], [0]])
    q_lab = (NU @ DELTA - I) @ y
    # Transform this into the phi frame.
    return np.array(inv(PHI) @ inv(CHI) @ inv(ETA) @ inv(MU) @ q_lab)
