"""Collection of auxiliary mathematical methods."""
from math import acos, cos, degrees, isclose, radians, sin
from typing import Any, List, Sequence, Tuple

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

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
    rot: Rotation = Rotation.from_rotvec(angle * np.array(axis) / norm(np.array(axis)))
    rot_as_matrix: np.ndarray = rot.as_matrix()
    return rot_as_matrix


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
        Angle between the vectors in degrees.
    """
    try:
        costheta = dot3(x * (1 / norm(x)), y * (1 / norm(y)))
    except ZeroDivisionError:
        return float("nan")
    return degrees(acos(bound(costheta)))


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


def angles_equivalent(first: float, second: float, tolerance: float = SMALL) -> bool:
    """Check for angle equivalence in degrees.

    Parameters
    ----------
    first: float
        First angle value in degrees.
    second:float
        Second angle value in degrees.
    tolerance: float, default = SMALL
        Absolute tolerance for the angle difference.

    Returns
    -------
    bool
        True is angles are equivalent.

    """
    diff = sin(radians(first - second) / 2.0)
    return is_small(diff, radians(tolerance))


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
    vector_norm: float = float(norm(vector))
    try:
        return vector * (1.0 / vector_norm)
    except ZeroDivisionError:
        return vector


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


def solve_h_fixed_q(
    h: float,
    qval: float,
    B: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
) -> List[Tuple[float, float, float]]:
    """Find valid hkl for a given h and q value.

    Coefficients are used to constrain solutions as:
            a*h + b*k + c*l = d

    Parameters
    ----------
    h : float
        value of h to use
    qval : float
        norm of the scattering vector squared
    B : np.ndarray
        3x3 matrix, usually the UB matrix
    a : float
        a coefficient to constrain the resulting hkl
    b : float
        a coefficient to constrain the resulting hkl
    c : float
        a coefficient to constrain the resulting hkl
    d : float
        a coefficient to constrain the resulting hkl

    Returns
    -------
    List[Tuple[float, float, float]]
        list of possible hkl solutions

    Raises
    ------
    DiffcalcException
        If the divisor is 0, or the discriminant is negative. The first of these occurs
        if both b and c are equal to 0.
    """
    B00, B10, B20 = B[0, 0], B[1, 0], B[2, 0]
    B01, B11, B21 = B[0, 1], B[1, 1], B[2, 1]
    B02, B12, B22 = B[0, 2], B[1, 2], B[2, 2]

    divisor = (
        B[:, 2].dot(B[:, 2]) * b**2
        - 2 * B[:, 1].dot(B[:, 2]) * b * c
        + B[:, 1].dot(B[:, 1]) * c**2
    )

    if divisor == 0.0:
        raise DiffcalcException("At least one of b or c must be non-zero")

    discriminant = (
        -(  # REPEATED 1.
            B02**2 * B11**2
            - 2 * B01 * B02 * B11 * B12
            + B01**2 * B12**2
            + (B02**2 + B12**2) * B21**2
            - 2 * (B01 * B02 + B11 * B12) * B21 * B22
            + (B01**2 + B11**2) * B22**2
        )
        * d**2
        + 2
        * (
            (  # REPEATED 1.
                B02**2 * B11**2
                - 2 * B01 * B02 * B11 * B12
                + B01**2 * B12**2
                + (B02**2 + B12**2) * B21**2
                - 2 * (B01 * B02 + B11 * B12) * B21 * B22
                + (B01**2 + B11**2) * B22**2
            )
            * a
            - (  # REPEATED 2.
                B02**2 * B10 * B11
                + B00 * B01 * B12**2
                + (B02**2 + B12**2) * B20 * B21
                + (B00 * B01 + B10 * B11) * B22**2
                - (B01 * B02 * B10 + B00 * B02 * B11) * B12
                - ((B01 * B02 + B11 * B12) * B20 + (B00 * B02 + B10 * B12) * B21) * B22
            )
            * b
            + (  # REPEATED 3.
                B01 * B02 * B10 * B11
                - B00 * B02 * B11**2
                + (B01 * B02 + B11 * B12) * B20 * B21
                - (B00 * B02 + B10 * B12) * B21**2
                - (B01**2 * B10 - B00 * B01 * B11) * B12
                - ((B01**2 + B11**2) * B20 - (B00 * B01 + B10 * B11) * B21) * B22
            )
            * c
        )
        * d
        * h
        - (
            (
                B02**2 * B11**2
                - 2 * B01 * B02 * B11 * B12
                + B01**2 * B12**2
                + (B02**2 + B12**2) * B21**2
                - 2 * (B01 * B02 + B11 * B12) * B21 * B22
                + (B01**2 + B11**2) * B22**2
            )
            * a**2
            - 2
            * (  # REPEATED 2.
                B02**2 * B10 * B11
                + B00 * B01 * B12**2
                + (B02**2 + B12**2) * B20 * B21
                + (B00 * B01 + B10 * B11) * B22**2
                - (B01 * B02 * B10 + B00 * B02 * B11) * B12
                - ((B01 * B02 + B11 * B12) * B20 + (B00 * B02 + B10 * B12) * B21) * B22
            )
            * a
            * b
            + (
                B02**2 * B10**2
                - 2 * B00 * B02 * B10 * B12
                + B00**2 * B12**2
                + (B02**2 + B12**2) * B20**2
                - 2 * (B00 * B02 + B10 * B12) * B20 * B22
                + (B00**2 + B10**2) * B22**2
            )
            * b**2
            + (
                B01**2 * B10**2
                - 2 * B00 * B01 * B10 * B11
                + B00**2 * B11**2
                + (B01**2 + B11**2) * B20**2
                - 2 * (B00 * B01 + B10 * B11) * B20 * B21
                + (B00**2 + B10**2) * B21**2
            )
            * c**2
            + 2
            * (
                (  # REPEATED 3.
                    B01 * B02 * B10 * B11
                    - B00 * B02 * B11**2
                    + (B01 * B02 + B11 * B12) * B20 * B21
                    - (B00 * B02 + B10 * B12) * B21**2
                    - (B01**2 * B10 - B00 * B01 * B11) * B12
                    - ((B01**2 + B11**2) * B20 - (B00 * B01 + B10 * B11) * B21)
                    * B22
                )
                * a
                - (
                    B01 * B02 * B10**2
                    - B00 * B02 * B10 * B11
                    + (B01 * B02 + B11 * B12) * B20**2
                    - (B00 * B02 + B10 * B12) * B20 * B21
                    - (B00 * B01 * B10 - B00**2 * B11) * B12
                    - ((B00 * B01 + B10 * B11) * B20 - (B00**2 + B10**2) * B21)
                    * B22
                )
                * b
            )
            * c
        )
        * h**2
        + divisor * qval
    )

    if discriminant < 0:
        raise DiffcalcException("No real solutions with given constraints.")

    if b != 0:
        coefficient = (B[:, 1].dot(B[:, 2]) * b - B[:, 1].dot(B[:, 1]) * c) * d - (
            (B[:, 1].dot(B[:, 2])) * a * b
            - (B[:, 0].dot(B[:, 2])) * b**2
            - (B[:, 1].dot(B[:, 1]) * a - B[:, 0].dot(B[:, 1]) * b) * c
        ) * h

        l1 = -(coefficient + np.sqrt(discriminant) * b) / divisor
        l2 = -(coefficient - np.sqrt(discriminant) * b) / divisor
        k1 = (d - a * h - c * l1) / b
        k2 = (d - a * h - c * l2) / b

        return [(h, k1, l1), (h, k2, l2)]

    else:
        coefficient = (B[:, 2].dot(B[:, 2]) * b - B[:, 1].dot(B[:, 2]) * c) * d - (
            (B[:, 2].dot(B[:, 2])) * a * b
            + (B[:, 0].dot(B[:, 1])) * c**2
            - (B[:, 1].dot(B[:, 2]) * a + B[:, 0].dot(B[:, 2]) * b) * c
        ) * h

        k1 = (coefficient - np.sqrt(discriminant) * c) / divisor
        k2 = (coefficient + np.sqrt(discriminant) * c) / divisor
        l1 = (d - a * h - b * k1) / c
        l2 = (d - a * h - b * k2) / c

        return [(h, k1, l1), (h, k2, l2)]


def solve_k_fixed_q(
    k: float,
    qval: float,
    B: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
) -> List[Tuple[float, float, float]]:
    """Find valid hkl for a given k and q value.

    Coefficients are used to constrain solutions as:
            a*h + b*k + c*l = d

    Parameters
    ----------
    k : float
        value of k to use
    qval : float
        norm of the scattering vector squared
    B : np.ndarray
        3x3 matrix, usually the UB matrix
    a : float
        a coefficient to constrain the resulting hkl
    b : float
        a coefficient to constrain the resulting hkl
    c : float
        a coefficient to constrain the resulting hkl
    d : float
        a coefficient to constrain the resulting hkl

    Returns
    -------
    List[Tuple[float, float, float]]
        list of possible hkl solutions

    Raises
    ------
    DiffcalcException
        If the divisor is 0, or the discriminant is negative. The first of these occurs
        if both a and c are equal to 0.
    """
    B00, B10, B20 = B[0, 0], B[1, 0], B[2, 0]
    B01, B11, B21 = B[0, 1], B[1, 1], B[2, 1]
    B02, B12, B22 = B[0, 2], B[1, 2], B[2, 2]

    divisor = (
        B[:, 2].dot(B[:, 2]) * a**2
        - 2 * B[:, 0].dot(B[:, 2]) * a * c
        + B[:, 0].dot(B[:, 0]) * c**2
    )

    if divisor == 0.0:
        raise DiffcalcException("At least one of a or c must be non-zero")

    discriminant = (
        -(
            B02**2 * B10**2
            - 2 * B00 * B02 * B10 * B12
            + B00**2 * B12**2
            + (B02**2 + B12**2) * B20**2
            - 2 * (B00 * B02 + B10 * B12) * B20 * B22
            + (B00**2 + B10**2) * B22**2
        )
        * d**2
        - 2
        * (
            (
                B02**2 * B10 * B11
                + B00 * B01 * B12**2
                + (B02**2 + B12**2) * B20 * B21
                + (B00 * B01 + B10 * B11) * B22**2
                - (B01 * B02 * B10 + B00 * B02 * B11) * B12
                - ((B01 * B02 + B11 * B12) * B20 + (B00 * B02 + B10 * B12) * B21) * B22
            )
            * a
            - (
                B02**2 * B10**2
                - 2 * B00 * B02 * B10 * B12
                + B00**2 * B12**2
                + (B02**2 + B12**2) * B20**2
                - 2 * (B00 * B02 + B10 * B12) * B20 * B22
                + (B00**2 + B10**2) * B22**2
            )
            * b
            + (
                B01 * B02 * B10**2
                - B00 * B02 * B10 * B11
                + (B01 * B02 + B11 * B12) * B20**2
                - (B00 * B02 + B10 * B12) * B20 * B21
                - (B00 * B01 * B10 - B00**2 * B11) * B12
                - ((B00 * B01 + B10 * B11) * B20 - (B00**2 + B10**2) * B21) * B22
            )
            * c
        )
        * d
        * k
        - (
            (
                B02**2 * B11**2
                - 2 * B01 * B02 * B11 * B12
                + B01**2 * B12**2
                + (B02**2 + B12**2) * B21**2
                - 2 * (B01 * B02 + B11 * B12) * B21 * B22
                + (B01**2 + B11**2) * B22**2
            )
            * a**2
            - 2
            * (
                B02**2 * B10 * B11
                + B00 * B01 * B12**2
                + (B02**2 + B12**2) * B20 * B21
                + (B00 * B01 + B10 * B11) * B22**2
                - (B01 * B02 * B10 + B00 * B02 * B11) * B12
                - ((B01 * B02 + B11 * B12) * B20 + (B00 * B02 + B10 * B12) * B21) * B22
            )
            * a
            * b
            + (
                B02**2 * B10**2
                - 2 * B00 * B02 * B10 * B12
                + B00**2 * B12**2
                + (B02**2 + B12**2) * B20**2
                - 2 * (B00 * B02 + B10 * B12) * B20 * B22
                + (B00**2 + B10**2) * B22**2
            )
            * b**2
            + (
                B01**2 * B10**2
                - 2 * B00 * B01 * B10 * B11
                + B00**2 * B11**2
                + (B01**2 + B11**2) * B20**2
                - 2 * (B00 * B01 + B10 * B11) * B20 * B21
                + (B00**2 + B10**2) * B21**2
            )
            * c**2
            + 2
            * (
                (
                    B01 * B02 * B10 * B11
                    - B00 * B02 * B11**2
                    + (B01 * B02 + B11 * B12) * B20 * B21
                    - (B00 * B02 + B10 * B12) * B21**2
                    - (B01**2 * B10 - B00 * B01 * B11) * B12
                    - ((B01**2 + B11**2) * B20 - (B00 * B01 + B10 * B11) * B21)
                    * B22
                )
                * a
                - (
                    B01 * B02 * B10**2
                    - B00 * B02 * B10 * B11
                    + (B01 * B02 + B11 * B12) * B20**2
                    - (B00 * B02 + B10 * B12) * B20 * B21
                    - (B00 * B01 * B10 - B00**2 * B11) * B12
                    - ((B00 * B01 + B10 * B11) * B20 - (B00**2 + B10**2) * B21)
                    * B22
                )
                * b
            )
            * c
        )
        * k**2
        + divisor * qval
    )

    if discriminant < 0:
        raise DiffcalcException("No real solutions with given constraints.")

    if a != 0:
        coefficient = ((B[:, 0].dot(B[:, 2])) * a - (B[:, 0].dot(B[:, 0])) * c) * d + (
            (B[:, 1].dot(B[:, 2])) * a**2
            - (B[:, 0].dot(B[:, 2])) * a * b
            - ((B[:, 0].dot(B[:, 1])) * a - (B[:, 0].dot(B[:, 0])) * b) * c
        ) * k

        l1 = -(coefficient + np.sqrt(discriminant) * a) / divisor
        l2 = -(coefficient - np.sqrt(discriminant) * a) / divisor
        h1 = (d - b * k - c * l1) / a
        h2 = (d - b * k - c * l2) / a

        return [(h1, k, l1), (h2, k, l2)]

    else:
        coefficient = ((B[:, 2].dot(B[:, 2])) * a - (B[:, 0].dot(B[:, 2])) * c) * d - (
            (B[:, 2].dot(B[:, 2])) * a * b
            + (B[:, 0].dot(B[:, 1])) * c**2
            - ((B[:, 1].dot(B[:, 2])) * a + (B[:, 0].dot(B[:, 2])) * b) * c
        ) * k

        h1 = (coefficient - np.sqrt(discriminant) * c) / divisor
        h2 = (coefficient + np.sqrt(discriminant) * c) / divisor
        l1 = (d - a * h1 - b * k) / c
        l2 = (d - a * h2 - b * k) / c

        return [(h1, k, l1), (h2, k, l2)]


def solve_l_fixed_q(
    l: float, qval: float, B: np.ndarray, a: float, b: float, c: float, d: float
) -> List[Tuple[float, float, float]]:
    """Find valid hkl for a given l and q value.

    Coefficients are used to constrain solutions as:
            a*h + b*k + c*l = d

    Parameters
    ----------
    l : float
        value of l to use
    qval : float
        norm of the scattering vector squared
    B : np.ndarray
        3x3 matrix, usually the UB matrix
    a : float
        a coefficient to constrain the resulting hkl
    b : float
        a coefficient to constrain the resulting hkl
    c : float
        a coefficient to constrain the resulting hkl
    d : float
        a coefficient to constrain the resulting hkl

    Returns
    -------
    List[Tuple[float, float, float]]
        list of possible hkl solutions

    Raises
    ------
    DiffcalcException
        If the divisor is 0, or the discriminant is negative. The first of these occurs
        if both a and b are equal to 0.
    """
    B00, B10, B20 = B[0, 0], B[1, 0], B[2, 0]
    B01, B11, B21 = B[0, 1], B[1, 1], B[2, 1]
    B02, B12, B22 = B[0, 2], B[1, 2], B[2, 2]

    divisor = (
        (B01**2 + B11**2 + B21**2) * a**2
        - 2 * (B00 * B01 + B10 * B11 + B20 * B21) * a * b
        + (B00**2 + B10**2 + B20**2) * b**2
    )

    if divisor == 0.0:
        raise DiffcalcException("At least one of b or c must be non-zero")

    discriminant = (
        -(
            B01**2 * B10**2
            - 2 * B00 * B01 * B10 * B11
            + B00**2 * B11**2
            + (B01**2 + B11**2) * B20**2
            - 2 * (B00 * B01 + B10 * B11) * B20 * B21
            + (B00**2 + B10**2) * B21**2
        )
        * d**2
        + 2
        * (
            (
                B01 * B02 * B10 * B11
                - B00 * B02 * B11**2
                + (B01 * B02 + B11 * B12) * B20 * B21
                - (B00 * B02 + B10 * B12) * B21**2
                - (B01**2 * B10 - B00 * B01 * B11) * B12
                - ((B01**2 + B11**2) * B20 - (B00 * B01 + B10 * B11) * B21) * B22
            )
            * a
            - (
                B01 * B02 * B10**2
                - B00 * B02 * B10 * B11
                + (B01 * B02 + B11 * B12) * B20**2
                - (B00 * B02 + B10 * B12) * B20 * B21
                - (B00 * B01 * B10 - B00**2 * B11) * B12
                - ((B00 * B01 + B10 * B11) * B20 - (B00**2 + B10**2) * B21) * B22
            )
            * b
            + (
                B01**2 * B10**2
                - 2 * B00 * B01 * B10 * B11
                + B00**2 * B11**2
                + (B01**2 + B11**2) * B20**2
                - 2 * (B00 * B01 + B10 * B11) * B20 * B21
                + (B00**2 + B10**2) * B21**2
            )
            * c
        )
        * d
        * l
        - (
            (
                B02**2 * B11**2
                - 2 * B01 * B02 * B11 * B12
                + B01**2 * B12**2
                + (B02**2 + B12**2) * B21**2
                - 2 * (B01 * B02 + B11 * B12) * B21 * B22
                + (B01**2 + B11**2) * B22**2
            )
            * a**2
            - 2
            * (
                B02**2 * B10 * B11
                + B00 * B01 * B12**2
                + (B02**2 + B12**2) * B20 * B21
                + (B00 * B01 + B10 * B11) * B22**2
                - (B01 * B02 * B10 + B00 * B02 * B11) * B12
                - ((B01 * B02 + B11 * B12) * B20 + (B00 * B02 + B10 * B12) * B21) * B22
            )
            * a
            * b
            + (
                B02**2 * B10**2
                - 2 * B00 * B02 * B10 * B12
                + B00**2 * B12**2
                + (B02**2 + B12**2) * B20**2
                - 2 * (B00 * B02 + B10 * B12) * B20 * B22
                + (B00**2 + B10**2) * B22**2
            )
            * b**2
            + (
                B01**2 * B10**2
                - 2 * B00 * B01 * B10 * B11
                + B00**2 * B11**2
                + (B01**2 + B11**2) * B20**2
                - 2 * (B00 * B01 + B10 * B11) * B20 * B21
                + (B00**2 + B10**2) * B21**2
            )
            * c**2
            + 2
            * (
                (
                    B01 * B02 * B10 * B11
                    - B00 * B02 * B11**2
                    + (B01 * B02 + B11 * B12) * B20 * B21
                    - (B00 * B02 + B10 * B12) * B21**2
                    - (B01**2 * B10 - B00 * B01 * B11) * B12
                    - ((B01**2 + B11**2) * B20 - (B00 * B01 + B10 * B11) * B21)
                    * B22
                )
                * a
                - (
                    B01 * B02 * B10**2
                    - B00 * B02 * B10 * B11
                    + (B01 * B02 + B11 * B12) * B20**2
                    - (B00 * B02 + B10 * B12) * B20 * B21
                    - (B00 * B01 * B10 - B00**2 * B11) * B12
                    - ((B00 * B01 + B10 * B11) * B20 - (B00**2 + B10**2) * B21)
                    * B22
                )
                * b
            )
            * c
        )
        * l**2
        + divisor * qval
    )

    if discriminant < 0:
        raise DiffcalcException("No real solutions with given constraints.")

    if a != 0:
        coefficient = (
            (B00 * B01 + B10 * B11 + B20 * B21) * a
            - (B00**2 + B10**2 + B20**2) * b
        ) * d + (
            (B01 * B02 + B11 * B12 + B21 * B22) * a**2
            - (B00 * B02 + B10 * B12 + B20 * B22) * a * b
            - (
                (B00 * B01 + B10 * B11 + B20 * B21) * a
                - (B00**2 + B10**2 + B20**2) * b
            )
            * c
        ) * l

        k1 = -(coefficient + np.sqrt(discriminant) * a) / divisor
        k2 = -(coefficient - np.sqrt(discriminant) * a) / divisor
        h1 = (d - b * k1 - c * l) / a
        h2 = (d - b * k2 - c * l) / a

        return [(h1, k1, l), (h2, k2, l)]

    else:
        coefficient = (
            (B01**2 + B11**2 + B21**2) * a
            - (B00 * B01 + B10 * B11 + B20 * B21) * b
        ) * d + (
            (B01 * B02 + B11 * B12 + B21 * B22) * a * b
            - (B00 * B02 + B10 * B12 + B20 * B22) * b**2
            - (
                (B01**2 + B11**2 + B21**2) * a
                - (B00 * B01 + B10 * B11 + B20 * B21) * b
            )
            * c
        ) * l

        h1 = (coefficient - np.sqrt(discriminant) * b) / divisor
        h2 = (coefficient + np.sqrt(discriminant) * b) / divisor
        k1 = (d - a * h1 - c * l) / b
        k2 = (d - a * h2 - c * l) / b

        return [(h1, k1, l), (h2, k2, l)]
