"""Collection of auxiliary mathematical methods."""
from math import acos, cos, isclose, sin
from typing import Any, Sequence, Tuple

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform.rotation import Rotation

I: np.ndarray = np.identity(3)

SMALL: float = 1e-7


def x_rotation(th: float) -> np.ndarray:
    """Rotation matrix over x axis.

    Parameters
    ----------
    th: float
        Rotation angle.

    Returns
    -------
    np.ndarray
        Rotation matrix.
    """
    return np.array(((1, 0, 0), (0, cos(th), -sin(th)), (0, sin(th), cos(th))))


def y_rotation(th: float) -> np.ndarray:
    """Rotation matrix over y axis.

    Parameters
    ----------
    th: float
        Rotation angle.

    Returns
    -------
    np.ndarray
        Rotation matrix.
    """
    return np.array(((cos(th), 0, sin(th)), (0, 1, 0), (-sin(th), 0, cos(th))))


def z_rotation(th: float) -> np.ndarray:
    """Rotation matrix over z axis.

    Parameters
    ----------
    th: float
        Rotation angle.

    Returns
    -------
    np.ndarray
        Rotation matrix.
    """
    return np.array(((cos(th), -sin(th), 0), (sin(th), cos(th), 0), (0, 0, 1)))


def xyz_rotation(axis: Tuple[float, float, float], angle: float) -> np.ndarray:
    """Rotation matrix over arbitrary axis.

    Parameters
    ----------
    axis: Tuple[float, float, float]
        Rotation axis coordinates
    angle: float
        Rotation angle.

    Returns
    -------
    np.ndarray
        Rotation matrix.
    """
    rot = Rotation.from_rotvec(angle * np.array(axis) / norm(np.array(axis)))
    return rot.as_matrix()


class DiffcalcException(Exception):
    """Error caused by user misuse of diffraction calculator."""

    def __str__(self):
        """Error message."""
        lines = []
        message = super().__str__()
        for msg_line in message.split("\n"):
            lines.append("* " + msg_line)
        width = max(len(line) for line in lines)
        lines.insert(0, "\n\n" + "*" * width)
        lines.append("*" * width)
        return "\n".join(lines)


### Matrices


def cross3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cross product of column vectors.

    Parameters
    ----------
    x: np.ndarray
        Column vector represented as NumPy (3,1) array.
    y: np.ndarray
        Column vector represented as NumPy (3,1) array.

    Returns
    -------
    np.ndarray
        Cross product column vector as as NumPy (3,1) array.
    """
    v = np.cross(x.T[0], y.T[0])
    return np.array([v]).T


def dot3(x: np.ndarray, y: np.ndarray) -> float:
    """Dot product of column vectors.

    Parameters
    ----------
    x: np.ndarray
        Column vector represented as NumPy (3,1) array.
    y: np.ndarray
        Column vector represented as NumPy (3,1) array.

    Returns
    -------
    float
        Dot product of column vectors.
    """
    return float(np.dot(x.T[0], y.T[0]))


def angle_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Angle between two column vectors.

    Parameters
    ----------
    x: np.ndarray
        Column vector represented as NumPy (3,1) array.
    y: np.ndarray
        Column vector represented as NumPy (3,1) array.

    Returns
    -------
    float
        Angle between the vectors.
    """
    costheta = dot3(x * (1 / norm(x)), y * (1 / norm(y)))
    return acos(bound(costheta))


## Math


def bound(x: float) -> float:
    """Check the input value is in [-1, 1] range.

    Rounds input value to +/-1 if |x| - 1 < SMALL.

    Parameters
    ----------
    x: float
        Input value to be checked

    Returns
    -------
    float
        Value in [-1, 1] range.

    Raises
    ------
    AssertionError
        Input value outside [-1, 1] range.
    """
    if abs(x) > (1 + SMALL):
        raise AssertionError(
            "The value (%f) was unexpectedly too far outside -1 or 1 to "
            "safely bound. Please report this." % x
        )
    if x > 1:
        return 1
    if x < -1:
        return -1
    return x


def radians_equivalent(first: float, second: float, tolerance: float = SMALL) -> bool:
    """Check for angle equivalence.

    Parameters
    ----------
    first: float
        First angle value.
    second:float
        Second angle value.
    tolerance: float, default = SMALL
        Absolute tolerance for the angle difference.

    Returns
    -------
    bool
        True is angles are equivalent.

    """
    diff = sin((first - second) / 2.0)
    return is_small(diff, tolerance)


def isnum(o: Any) -> bool:
    """Check if the input object type is either int or float.

    Parameters
    ----------
    o: Any
        Input object to be checked.

    Returns
    -------
    bool
        If object type is either int or float.
    """
    return isinstance(o, (int, float))


def allnum(lst: Sequence[Any]) -> bool:
    """Check if all object types in the input sequence are either int or float.

    Parameters
    ----------
    o: Sequence[Any]
        Input object sequence to be checked.

    Returns
    -------
    bool
        If all object types in th sequence are either int or float.
    """
    return not [o for o in lst if not isnum(o)]


def is_small(x, tolerance=SMALL) -> bool:
    """Check if input value is 0 within tolerance.

    Parameters
    ----------
    x: float
        Input value to be checked.
    tolerance: float, default = SMALL
        Absolute tolerance.

    Returns
    -------
    bool
        True is the value is 0 within tolerance.
    """
    return isclose(x, 0, abs_tol=tolerance)


def sign(x: float, tolerance: float = SMALL) -> int:
    """Sign function with specified tolerance.

    Parameters
    ----------
    x: float
        Function argument.
    tolerance: float, default = SMALL
        Absolute tolerance.

    Returns
    -------
    int
        1 for positive, -1 for negative values and
        0 if argument equals 0 within tolerance.
    """
    if is_small(x, tolerance):
        return 0
    if x > 0:
        return 1
    # x < 0
    return -1


def normalised(vector: np.ndarray) -> np.ndarray:
    """Normalise vector array.

    Return normalised vector coordinates.

    Parameters
    ----------
    vector : ndarray
        The vector to be normalised.

    Returns
    -------
    ndarray
        Normalised vector.
    """
    return vector * (1.0 / norm(vector))


def zero_round(num):
    """Round to zero if small.

    This is useful to get rid of erroneous minus signs
    resulting from float representation close to zero.

    Parameters
    ----------
    num : number
        The value to be checked for rounding.

    Returns
    -------
    number
        The rounded input value.
    """
    if abs(num) < SMALL:
        num = 0
    return num
