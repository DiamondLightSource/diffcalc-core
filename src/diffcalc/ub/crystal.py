"""Crystal lattice information.

A module defining crystal lattice class and auxiliary methods for calculating
crystal plane geometric properties.
"""
from math import acos, cos, pi, sin, sqrt
from typing import List, Optional, Tuple

import numpy as np
from diffcalc.util import TODEG, TORAD, allnum, angle_between_vectors, zero_round
from numpy.linalg import inv


class Crystal:
    """Class containing crystal lattice information and auxiliary routines.

    Contains the lattice parameters and calculated B matrix for the crystal
    under test. Also Calculates the distance between planes at a given hkl
    value.

    Attributes
    ----------
    name: str
        Crystal name.
    a1: float
        Crystal lattice parameter.
    a2: float
        Crystal lattice parameter.
    a3: float
        Crystal lattice parameter.
    alpha1: float
        Crystal lattice angle.
    alpha2: float
        Crystal lattice angle.
    alpha3: float
        Crystal lattice angle.
    system: str
        Crystal system name.
    B: np.ndarray
        B matrix.
    """

    def __init__(
        self,
        name: str,
        system: Optional[str] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        """Create a new crystal lattice and calculates B matrix.

        Parameters
        ----------
        name: str
            Crystal name
        system: Optional[float], default = None
            Crystal lattice type.
        a: Optional[float], default = None
            Crystal lattice parameter.
        b: Optional[float], default = None
            Crystal lattice parameter.
        c: Optional[float], default = None
            Crystal lattice parameter.
        alpha: Optional[float], default = None
            Crystal lattice angle.
        beta: Optional[float], default = None
            Crystal lattice angle.
        gamma: Optional[float], default = None
            Crystal lattice angle.
        """
        self.name = name
        args = tuple(
            val for val in (system, a, b, c, alpha, beta, gamma) if val is not None
        )
        if allnum(args):
            if len(args) != 6:
                raise ValueError(
                    "Crystal definition requires six lattice "
                    "parameters or crystal system name."
                )
            # Set the direct lattice parameters
            self.system = "Triclinic"
            self.a1, self.a2, self.a3 = tuple(float(val) for val in args[:3])
            self.alpha1, self.alpha2, self.alpha3 = tuple(
                float(val) * TORAD for val in args[3:]
            )
            self._set_reciprocal_cell(
                self.a1, self.a2, self.a3, self.alpha1, self.alpha2, self.alpha3
            )
        else:
            if not isinstance(args[0], str):
                raise ValueError(f"Invalid crystal system name {args[0]}.")
            self.system = args[0]
            if allnum(args[1:]):
                self._set_cell_for_system(system, a, b, c, alpha, beta, gamma)
            else:
                raise ValueError("Crystal lattice parameters must be numeric type.")

    def __str__(self) -> str:
        """Represent the crystal lattice information as a string.

        Retruns
        -------
        str
            Crystal lattice information string.
        """
        return "\n".join(self._str_lines())

    def _str_lines(self) -> List[str]:
        WIDTH = 13
        if self.name is None:
            return ["   none specified"]
        lines = []
        lines.append("   name:".ljust(WIDTH) + self.name.rjust(9))
        lines.append("")
        lines.append(
            "   a, b, c:".ljust(WIDTH)
            + "% 9.5f % 9.5f % 9.5f" % (self.get_lattice()[1:4])
        )
        lines.append(
            " " * WIDTH
            + "% 9.5f % 9.5f % 9.5f  %s" % (self.get_lattice()[4:] + (self.system,))
        )
        lines.append("")

        fmt = "% 9.5f % 9.5f % 9.5f"
        lines.append(
            "   B matrix:".ljust(WIDTH)
            + fmt
            % (
                zero_round(self.B[0, 0]),
                zero_round(self.B[0, 1]),
                zero_round(self.B[0, 2]),
            )
        )
        lines.append(
            " " * WIDTH
            + fmt
            % (
                zero_round(self.B[1, 0]),
                zero_round(self.B[1, 1]),
                zero_round(self.B[1, 2]),
            )
        )
        lines.append(
            " " * WIDTH
            + fmt
            % (
                zero_round(self.B[2, 0]),
                zero_round(self.B[2, 1]),
                zero_round(self.B[2, 2]),
            )
        )
        return lines

    def _set_reciprocal_cell(
        self,
        a1: float,
        a2: float,
        a3: float,
        alpha1: float,
        alpha2: float,
        alpha3: float,
    ) -> None:
        # Calculate the reciprocal lattice parameters
        beta2 = acos(
            (cos(alpha1) * cos(alpha3) - cos(alpha2)) / (sin(alpha1) * sin(alpha3))
        )

        beta3 = acos(
            (cos(alpha1) * cos(alpha2) - cos(alpha3)) / (sin(alpha1) * sin(alpha2))
        )

        volume = (
            a1
            * a2
            * a3
            * sqrt(
                1
                + 2 * cos(alpha1) * cos(alpha2) * cos(alpha3)
                - cos(alpha1) ** 2
                - cos(alpha2) ** 2
                - cos(alpha3) ** 2
            )
        )

        b1 = 2 * pi * a2 * a3 * sin(alpha1) / volume
        b2 = 2 * pi * a1 * a3 * sin(alpha2) / volume
        b3 = 2 * pi * a1 * a2 * sin(alpha3) / volume

        # Calculate the BMatrix from the direct and reciprical parameters.
        # Reference: Busang and Levy (1967)
        self.B = np.array(
            [
                [b1, b2 * cos(beta3), b3 * cos(beta2)],
                [0.0, b2 * sin(beta3), -b3 * sin(beta2) * cos(alpha1)],
                [0.0, 0.0, 2 * pi / a3],
            ]
        )

    def get_lattice(self) -> Tuple[str, float, float, float, float, float, float]:
        """Get all crystal name and crystal lattice parameters.

        Returns
        -------
        Tuple[str, float, float, float, float, float, float]
            Crystal name and crystal lattice parameters.
        """
        return (
            self.name,
            self.a1,
            self.a2,
            self.a3,
            self.alpha1 * TODEG,
            self.alpha2 * TODEG,
            self.alpha3 * TODEG,
        )

    def get_lattice_params(self) -> Tuple[str, Tuple[float, ...]]:
        """Get crystal name and non-redundant set of crystal lattice parameters.

        Returns
        -------
        Tuple[str, Tuple[float, ...]]
            Crystal name and minimal set of parameters for the crystal lattice
            system.
        """
        try:
            if self.system == "Triclinic":
                return self.system, (
                    self.a1,
                    self.a2,
                    self.a3,
                    self.alpha1 * TODEG,
                    self.alpha2 * TODEG,
                    self.alpha3 * TODEG,
                )
            elif self.system == "Monoclinic":
                return self.system, (
                    self.a1,
                    self.a2,
                    self.a3,
                    self.alpha2 * TODEG,
                )
            elif self.system == "Orthorhombic":
                return self.system, (self.a1, self.a2, self.a3)
            elif self.system == "Tetragonal" or self.system == "Hexagonal":
                return self.system, (self.a1, self.a3)
            elif self.system == "Rhombohedral":
                return self.system, (self.a1, self.alpha1 * TODEG)
            elif self.system == "Cubic":
                return self.system, (self.a1,)
            else:
                raise TypeError(
                    "Invalid crystal system parameter: %s" % str(self.system)
                )
        except ValueError as e:
            raise TypeError from e

    def _get_cell_for_system(
        self, system: str
    ) -> Tuple[float, float, float, float, float, float]:
        if system == "Triclinic":
            return (
                self.a1,
                self.a2,
                self.a3,
                self.alpha1 * TORAD,
                self.alpha2 * TORAD,
                self.alpha3 * TORAD,
            )
        elif system == "Monoclinic":
            return (self.a1, self.a2, self.a3, pi / 2, self.alpha2 * TORAD, pi / 2)
        elif system == "Orthorhombic":
            return (self.a1, self.a2, self.a3, pi / 2, pi / 2, pi / 2)
        elif system == "Tetragonal":
            return (self.a1, self.a1, self.a3, pi / 2, pi / 2, pi / 2)
        elif system == "Rhombohedral":
            return (
                self.a1,
                self.a1,
                self.a1,
                self.alpha1 * TORAD,
                self.alpha1 * TORAD,
                self.alpha1 * TORAD,
            )
        elif system == "Hexagonal":
            return (self.a1, self.a1, self.a3, pi / 2, pi / 2, 2 * pi / 3)
        elif system == "Cubic":
            return (self.a1, self.a1, self.a1, pi / 2, pi / 2, pi / 2)
        else:
            raise TypeError("Invalid crystal system parameter: %s" % str(system))

    def _set_cell_for_system(
        self,
        system: str,
        a: float,
        b: Optional[float] = None,
        c: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        args = tuple(val for val in (a, b, c, alpha, beta, gamma) if val is not None)
        try:
            if len(args) == 6 or system == "Triclinic":
                (
                    self.a1,
                    self.a2,
                    self.a3,
                    self.alpha1,
                    self.alpha2,
                    self.alpha3,
                ) = args
            elif system == "Monoclinic":
                (self.a1, self.a2, self.a3, self.alpha2) = args
            elif system == "Orthorhombic":
                (self.a1, self.a2, self.a3) = args
            elif system == "Tetragonal" or system == "Hexagonal":
                (self.a1, self.a3) = args
            elif system == "Rhombohedral":
                (self.a1, self.alpha1) = args
            elif system == "Cubic":
                (self.a1,) = args
            else:
                raise TypeError("Invalid crystal system parameter: %s" % str(system))
        except ValueError as e:
            raise TypeError from e
        (
            self.a1,
            self.a2,
            self.a3,
            self.alpha1,
            self.alpha2,
            self.alpha3,
        ) = self._get_cell_for_system(system)
        self._set_reciprocal_cell(
            self.a1, self.a2, self.a3, self.alpha1, self.alpha2, self.alpha3
        )

    def get_hkl_plane_distance(self, hkl: Tuple[float, float, float]) -> float:
        """Calculate distance between crystal lattice planes.

        Parameters
        ----------
        hkl: Tuple[float, float, float]
            Miller indices of the lattice plane.

        Returns
        -------
        float
            Crystal lattice plane distance.
        """
        hkl_vector = np.array([hkl])
        b_reduced = self.B / (2 * pi)
        bMT = inv(b_reduced) @ inv(b_reduced.T)
        return 1.0 / sqrt((hkl_vector @ inv(bMT) @ hkl_vector.T)[0, 0])

    def get_hkl_plane_angle(
        self, hkl1: Tuple[float, float, float], hkl2: Tuple[float, float, float]
    ) -> float:
        """Calculate the angle between crystal lattice planes.

        Parameters
        ----------
        hkl1: Tuple[float, float, float]
            Miller indices of the first lattice plane
        hkl2: Tuple[float, float, float]
            Miller indices of the second lattice plane

        Returns
        -------
        float
            The angle between the crystal lattice planes.
        """
        hkl1 = np.array([hkl1]).T
        hkl2 = np.array([hkl2]).T
        nphi1 = self.B @ hkl1
        nphi2 = self.B @ hkl2
        angle = angle_between_vectors(nphi1, nphi2)
        return angle
