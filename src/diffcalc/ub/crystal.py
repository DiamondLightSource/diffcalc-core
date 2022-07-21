"""Crystal lattice information.

A module defining crystal lattice class and auxiliary methods for calculating
crystal plane geometric properties.
"""
from dataclasses import dataclass
from math import acos, cos, degrees, pi, radians, sin, sqrt
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from diffcalc.ub.systems import Systems, SystemType, available_systems
from diffcalc.util import DiffcalcException, angle_between_vectors, zero_round
from numpy.linalg import inv


def lists_equal(list1: List[Any], list2: List[Any]) -> bool:
    if len(list1) != len(list2):
        return False
    return bool(np.all([item in list2 for item in list1]))


@dataclass
class JSONCrystal:
    name: str
    system: str
    lattice_params: Dict[str, float]

    @property
    def asdict(self):
        return self.__dict__


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

    mapping = {
        "a": "a1",
        "b": "a2",
        "c": "a3",
        "alpha": "alpha1",
        "beta": "alpha2",
        "gamma": "alpha3",
    }

    def __init__(
        self,
        name: str,
        system: str,
        lattice_params: Union[Dict[str, float], List[float]],
        indegrees: bool = True,
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
        self.indegrees = indegrees
        self.default_params: SystemType = Systems[system].value.copy()

        self.a1, self.a2, self.a3 = 0.0, 0.0, 0.0
        self.alpha1, self.alpha2, self.alpha3 = 0.0, 0.0, 0.0

        if system in available_systems:

            required_params = [
                key for key, item in self.default_params.items() if item is None
            ]
            if not isinstance(lattice_params, dict):
                lattice_params = dict(zip(required_params, lattice_params))

            self.lattice_params = lattice_params

            params_equal = lists_equal(required_params, list(lattice_params.keys()))
            if not params_equal:
                raise DiffcalcException(
                    f"incorrect parameters for {system} system. "
                    f"This requires: {required_params} as floating point values"
                )

            self.system = system
        else:
            raise DiffcalcException(f"system must be one of: {available_systems}")

        self._set_cell_for_system(lattice_params)

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
    ) -> None:
        a1, a2, a3 = self.a1, self.a2, self.a3
        alpha1, alpha2, alpha3 = self.alpha1, self.alpha2, self.alpha3

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
            degrees(self.alpha1),
            degrees(self.alpha2),
            degrees(self.alpha3),
        )

    def get_lattice_params(self) -> Tuple[str, Dict[str, float]]:
        """Get crystal name and non-redundant set of crystal lattice parameters.

        Returns
        -------
        Tuple[str, Tuple[float, ...]]
            Crystal name and minimal set of parameters for the crystal lattice
            system.
        """
        return self.system, self.lattice_params

    def _set_cell_for_system(self, params: Dict[str, float]) -> None:
        default_params = self.default_params

        for param_key, param_value in params.items():
            if (param_key == ("alpha" or "beta" or "gamma")) and self.indegrees:
                default_params[param_key] = radians(param_value)
            else:
                default_params[param_key] = param_value

        for default_key, default_value in default_params.items():
            if isinstance(default_value, str):
                default_params[default_key] = default_params[default_value]

        for key in default_params:
            attr = self.mapping[key]
            setattr(self, attr, default_params[key])

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
            Miller indices of the first lattice plane
        hkl2: Tuple[float, float, float]
            Miller indices of the second lattice plane

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
        """Serialise the crystal into a JSON compatible dictionary"""
        return {
            "name": self.name,
            "system": self.system,
            "lattice_params": self.lattice_params,
            "indegrees": self.indegrees,
        }

    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> "Crystal":
        return Crystal(**data)


# def deserialise_crystal(crystal_data: JSONCrystal):
#     return Crystal(**crystal_data.asdict)


# test = Crystal(
#     name="test", lattice_params={"a": 4.913, "c": 5.405}, system="Tetragonal"
# )
# test.get_hkl_plane_angle((0, 0, 1), (0, 1, 3))
# data = test.asdict

# output = test.get_lattice_params()
# print("yay")
