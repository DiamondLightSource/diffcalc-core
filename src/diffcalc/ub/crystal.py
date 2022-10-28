"""Crystal lattice information.

A module defining crystal lattice class and auxiliary methods for calculating
crystal plane geometric properties.
"""
from dataclasses import dataclass
from inspect import signature
from math import acos, cos, pi, sin, sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from diffcalc.util import Angle, DiffcalcException, angle_between_vectors, zero_round
from numpy.linalg import inv
from pint import Quantity


@dataclass
class LatticeParams:
    """Lattice parameters to be passed to Crystal class."""

    a: float
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: Optional[Angle] = None
    beta: Optional[Angle] = None
    gamma: Optional[Angle] = None


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
        self, name: str, params: LatticeParams, system: Optional[str] = None
    ) -> None:
        """Create a new crystal lattice and calculates B matrix.

        Parameters
        ----------
        name: str
            Crystal name.
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
        self.a = params.a
        self.b = params.b
        self.c = params.c
        self.alpha = params.alpha
        self.beta = params.beta
        self.gamma = params.gamma
        self.system = system

        self._set_cell_for_system()

    def __str__(self) -> str:
        """Represent the crystal lattice information as a string.

        Returns
        -------
        str
            Crystal lattice information string.
        """
        return "\n".join(self._str_lines())

    def _str_lines(self) -> List[str]:
        WIDTH = 13
        a, b, c, alpha, beta, gamma = self.get_lattice()
        if self.name is None:
            return ["   none specified"]
        lines = []
        lines.append("   name:".ljust(WIDTH) + self.name.rjust(9))
        lines.append("")
        lines.append("   a, b, c:".ljust(WIDTH) + f"{a: 9.5f} {b: 9.5f} {c: 9.5f}")

        angle_string = ""
        for angle in (alpha, beta, gamma):
            if isinstance(angle, Quantity):
                if angle.units == "degree":
                    angle_string += "% 9.5f " % angle.magnitude
            else:
                angle_string += "% 9.5f " % angle

        lines.append(" " * WIDTH + angle_string + " %s" % self.system)
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
    ) -> None:
        a1, a2, a3 = self.a, self.b, self.c
        alpha1, alpha2, alpha3 = self.alpha, self.beta, self.gamma

        if (sin(alpha1) == 0.0) or (sin(alpha2) == 0.0) or (sin(alpha3) == 0.0):
            raise DiffcalcException(
                "Error setting reciprocal cell: alpha, beta and "
                + "gamma cannot be multiples of 2 pi."
            )

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

        # Calculate the BMatrix from the direct and reciprocal parameters.
        # Reference: Busang and Levy (1967)
        self.B = np.array(
            [
                [b1, b2 * cos(beta3), b3 * cos(beta2)],
                [0.0, b2 * sin(beta3), -b3 * sin(beta2) * cos(alpha1)],
                [0.0, 0.0, 2 * pi / a3],
            ]
        )

    def get_lattice(self) -> Tuple[float, float, float, Angle, Angle, Angle]:
        """Get crystal lattice parameters.

        Returns
        -------
        Tuple[str, float, float, float, Angle, Angle, Angle]
            Crystal name and crystal lattice parameters.
        """
        return (
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        )

    def get_lattice_params(self) -> Tuple[Union[float, Angle], ...]:
        """Get non-redundant set of crystal lattice parameters.

        Returns
        -------
        Tuple[str, Tuple[float, ...]]
            minimal set of parameters for the crystal lattice system.
        """
        system_mappings: Dict[str, Tuple[Union[float, Angle], ...]] = {
            "Triclinic": (self.a, self.b, self.c, self.alpha, self.beta, self.gamma),
            "Monoclinic": (self.a, self.b, self.c, self.beta),
            "Orthorhombic": (self.a, self.b, self.c),
            "Tetragonal": (self.a, self.c),
            "Hexagonal": (self.a, self.c),
            "Rhombohedral": (self.a, self.alpha),
            "Cubic": (self.a,),
        }

        try:
            return system_mappings[self.system]
        except KeyError:
            raise DiffcalcException(
                f"Provided crystal system {self.system} is invalid. Please "
                + f"choose one of: {system_mappings.keys()}"
            )

    def _set_triclinic_cell(
        self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
    ) -> Tuple[float, float, float, float, float, float]:
        return a, b, c, alpha, beta, gamma

    def _set_monoclinic_cell(
        self, a: float, b: float, c: float, beta: float
    ) -> Tuple[float, float, float, float, float, float]:
        return a, b, c, pi / 2, beta, pi / 2

    def _set_orthorhombic_cell(
        self, a: float, b: float, c: float
    ) -> Tuple[float, float, float, float, float, float]:
        return a, b, c, pi / 2, pi / 2, pi / 2

    def _set_tetragonal_cell(
        self, a: float, c: float
    ) -> Tuple[float, float, float, float, float, float]:
        return a, a, c, pi / 2, pi / 2, pi / 2

    def _set_hexagonal_cell(
        self, a: float, c: float
    ) -> Tuple[float, float, float, float, float, float]:
        return a, a, c, pi / 2, pi / 2, 2 * pi / 3

    def _set_rhombohedral_cell(
        self, a: float, alpha: float
    ) -> Tuple[float, float, float, float, float, float]:
        return a, a, a, alpha, alpha, alpha

    def _set_cubic_cell(
        self, a: float
    ) -> Tuple[float, float, float, float, float, float]:
        return a, a, a, pi / 2, pi / 2, pi / 2

    def _set_cell_for_system(
        self,
    ) -> None:
        raw_input = [self.a, self.b, self.c, self.alpha, self.beta, self.gamma]
        params = [entry for entry in raw_input if entry is not None]

        system_mappings: Dict[
            str, Callable[..., Tuple[float, float, float, float, float, float]]
        ] = {
            "Monoclinic": self._set_monoclinic_cell,
            "Orthorhombic": self._set_orthorhombic_cell,
            "Hexagonal": self._set_hexagonal_cell,
            "Rhombohedral": self._set_rhombohedral_cell,
            "Tetragonal": self._set_tetragonal_cell,
            "Cubic": self._set_cubic_cell,
        }

        required_lengths = {
            len(signature(method).parameters): system
            for system, method in system_mappings.items()
        }
        required_lengths[6] = "Triclinic"

        system = self.system
        if system is None:
            try:
                system = required_lengths[len(params)]
            except KeyError:
                raise DiffcalcException(
                    f"{len(params)} were given, but no such system"
                    + "requires this many parameters. Required parameter lengths are "
                    + f"{required_lengths} for each system."
                )
        else:
            if system not in system_mappings.keys() and system != "Triclinic":
                raise DiffcalcException(
                    f"Provided crystal system {self.system} is invalid. Please "
                    + f"choose one of: {system_mappings.keys()}"
                )
            minimum_length = (
                len(signature(system_mappings[system]).parameters)
                if system != "Triclinic"
                else 6
            )

            if (len(params) != minimum_length) and (len(params) != 6):
                raise DiffcalcException(
                    "Parameters to construct the Crystal do not match specified system."
                    + f" {self.system} system requires either exactly {minimum_length} "
                    + f"parameter(s) or 6 but got {len(params)}."
                )

        if len(params) < 6:
            system_method = system_mappings[system]
            (a, b, c, alpha, beta, gamma) = system_method(*params)

            self.a = a
            self.b = b
            self.c = c
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

        self.system = system
        self._set_reciprocal_cell()

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
            Miller indices of the first lattice plane.
        hkl2: Tuple[float, float, float]
            Miller indices of the second lattice plane.

        Returns
        -------
        float
            The angle between the crystal lattice planes.
        """
        hkl1_transpose = np.array([hkl1]).T
        hkl2_transpose = np.array([hkl2]).T
        nphi1 = self.B @ hkl1_transpose
        nphi2 = self.B @ hkl2_transpose
        angle = angle_between_vectors(nphi1, nphi2)
        return angle

    @property
    def asdict(self) -> Dict[str, Any]:
        """Serialise the crystal into a JSON compatible dictionary.

        Note, because the class automatically assumes all angles are
        in degrees, the returned angles alpha, beta and gamma are given
        in degrees such that the dictionary can be directly unpacked as is.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing properties of crystal class. Can
            be directly unpacked to recreate Crystal object, i.e.
            Crystal(**returned_dict).

        """
        return {
            "name": self.name,
            "params": LatticeParams(
                self.a, self.b, self.c, self.alpha, self.beta, self.gamma
            ),
            "system": self.system,
        }
