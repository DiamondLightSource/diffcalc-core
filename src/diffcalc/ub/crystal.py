"""Crystal lattice information.

A module defining crystal lattice class and auxiliary methods for calculating
crystal plane geometric properties.
"""
from inspect import signature
from math import acos, cos, pi, sin, sqrt
from typing import Callable, Dict, List, Tuple

import numpy as np
from diffcalc.util import DiffcalcException, angle_between_vectors, zero_round
from numpy.linalg import inv
from typing_extensions import TypedDict


class Crystal(TypedDict):
    """Internal Crystal definition.

    Conversions between degrees/radians and other functions are managed by the
    CrystalHandler.
    """

    name: str
    system: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


class CrystalHandler:
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

    def __init__(self, name: str, params: List[float], system: str) -> None:
        """Create a new crystal lattice and calculates B matrix.

        Parameters
        ----------
        name: str
            Crystal name.
        params: List[float]
            List of lattice parameters. Can be any size as long as it is not empty.
        system: Optional[float], default = None
            Crystal lattice type.
        indegrees: bool = True
            are the crystal params are given in degrees. If this is true, calls to
            asdict will also return in degrees, not radians. All internal logic done
            in radians, this just changes how the user sees the inputs/outputs.

        """
        self.name = name
        self.system = system

        self._set_cell_for_system(params)

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
        if self.name is None:
            return ["   none specified"]
        lines = []
        lines.append("   name:".ljust(WIDTH) + self.name.rjust(9))
        lines.append("")
        lines.append(
            "   a, b, c:".ljust(WIDTH)
            + "% 9.5f % 9.5f % 9.5f" % (self.get_lattice()[2:5])
        )
        lines.append(
            " " * WIDTH
            + "% 9.5f % 9.5f % 9.5f  %s" % (self.get_lattice()[5:] + (self.system,))
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

    def _set_reciprocal_cell(self, a1, a2, a3, alpha1, alpha2, alpha3) -> None:
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
        self.B: np.ndarray = np.array(
            [
                [b1, b2 * cos(beta3), b3 * cos(beta2)],
                [0.0, b2 * sin(beta3), -b3 * sin(beta2) * cos(alpha1)],
                [0.0, 0.0, 2 * pi / a3],
            ]
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

    def get_lattice(self) -> Tuple[str, str, float, float, float, float, float, float]:
        """Get all crystal name and crystal lattice parameters.

        Returns
        -------
        Tuple[str, float, float, float, float, float, float]
            Crystal name and crystal lattice parameters.
        """
        return (
            self.name,
            self.system,
            self.crystal["a"],
            self.crystal["b"],
            self.crystal["c"],
            self.crystal["alpha"],
            self.crystal["beta"],
            self.crystal["gamma"],
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
                    self.crystal["a"],
                    self.crystal["b"],
                    self.crystal["c"],
                    self.crystal["alpha"],
                    self.crystal["beta"],
                    self.crystal["gamma"],
                )
            elif self.system == "Monoclinic":
                return self.system, (
                    self.crystal["a"],
                    self.crystal["b"],
                    self.crystal["c"],
                    self.crystal["beta"],
                )
            elif self.system == "Orthorhombic":
                return self.system, (
                    self.crystal["a"],
                    self.crystal["b"],
                    self.crystal["c"],
                )
            elif self.system == "Tetragonal" or self.system == "Hexagonal":
                return self.system, (self.crystal["a"], self.crystal["c"])
            elif self.system == "Rhombohedral":
                return self.system, (
                    self.crystal["a"],
                    self.crystal["alpha"],
                )
            elif self.system == "Cubic":
                return self.system, (self.crystal["a"],)
            else:
                raise TypeError(
                    "Invalid crystal system parameter: %s" % str(self.system)
                )
        except ValueError as e:
            raise TypeError from e

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

    def _set_cell_full_params(
        self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
    ):
        return a, b, c, alpha, beta, gamma

    def _set_cell_for_system(
        self,
        params: List[float],
    ) -> None:
        """Any angles are already in radians."""
        system_mappings: Dict[
            str, Callable[..., Tuple[float, float, float, float, float, float]]
        ] = {
            "Triclinic": self._set_triclinic_cell,
            "Monoclinic": self._set_monoclinic_cell,
            "Orthorhombic": self._set_orthorhombic_cell,
            "Tetragonal": self._set_tetragonal_cell,
            "Hexagonal": self._set_hexagonal_cell,
            "Rhombohedral": self._set_rhombohedral_cell,
            "Cubic": self._set_cubic_cell,
        }

        system_method = system_mappings[self.system]
        required_length = len(signature(system_method).parameters)

        if len(params) == required_length:
            (a, b, c, alpha, beta, gamma) = system_method(*params)
        elif len(params) == 6:
            (a, b, c, alpha, beta, gamma) = self._set_cell_full_params(*params)
        else:
            raise DiffcalcException(
                "Parameters to construct the Crystal do not match specified system."
                + f" {self.system} system requires exactly {required_length} "
                + f"or 6 parameter(s), but got {len(params)}."
            )

        self._set_reciprocal_cell(a, b, c, alpha, beta, gamma)
        self.crystal: Crystal = Crystal(
            name=self.name,
            system=self.system,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    @property
    def asdict(self) -> Crystal:
        """Serialise the crystal into a JSON compatible dictionary.

        Returns a typed dictionary containing all unit cell parameters.
        This will correctly return in either degrees or radians depending on
        self.indegrees.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing properties of crystal class. Can
            be directly unpacked to recreate Crystal object, i.e.
            Crystal(**returned_dict).

        """
        name, system, a, b, c, alpha, beta, gamma = self.get_lattice()
        return Crystal(
            name=name, system=system, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
        )

    @classmethod
    def fromdict(cls, crystal_info: Crystal) -> "CrystalHandler":
        """Deserialise a typed dict into a class object instance.

        Returns a new CrystalHandler instance containing all information from the
        typed dict. CrystalHandler automatically creates a crystal instance within
        itself that is in radians.

        Parameters
        ----------
        crystal_info: Crystal
            A typed dict containing basic crystal information.
        indegrees: bool
            Boolean to determine if the crystal_info is in degrees or not.

        Returns
        -------
        CrystalHandler
            Instance of this class.

        """
        params = [
            crystal_info["a"],
            crystal_info["b"],
            crystal_info["c"],
            crystal_info["alpha"],
            crystal_info["beta"],
            crystal_info["gamma"],
        ]

        return cls(crystal_info["name"], params, crystal_info["system"])
