"""Routines for UB matrix calculation.

This module provides a number of objects and functions for setting UB matrix
and reference/surface normal vectors based on reflections and/or orientations
and crystal miscut information.
"""
import pickle
import uuid
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from math import acos, asin, cos, pi, sin
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from diffcalc.hkl.geometry import Position, get_q_phi, get_rotation_matrices
from diffcalc.ub.crystal import Crystal
from diffcalc.ub.fitting import fit_crystal, fit_u_matrix
from diffcalc.ub.reference import OrientationList, Reflection, ReflectionList
from diffcalc.util import (
    SMALL,
    TODEG,
    TORAD,
    DiffcalcException,
    allnum,
    bound,
    cross3,
    dot3,
    is_small,
    xyz_rotation,
    zero_round,
)
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation


@dataclass
class ReferenceVector:
    """Class representing reference vector information.

    Reference vector object is used to define orientations
    in reciprocal and laboratory coordinate systems e.g. to
    represent an azimuthat direction or a surface normal.

    Attributes
    ----------
    n_ref: Tuple[float, float, float]
        tuple with vector coordinates
    rlv: bool
        Flag indicating if coordinates are in reciprocal or
        laboratory coordinate system.
        True: for reciprocal space vector.
        False: for real space vector.

    Methods
    -------
    get_array(UB: Optional[np.ndarray] = None) -> np.ndarray
        Return reference vector as 3x1 NumPy array
    set_array(n_ref: np.ndarray) -> Tuple[float, float, float]
        Set reference vector coordinates from 3x1 NumPy array
    """

    n_ref: Tuple[float, float, float]
    rlv: bool

    def get_array(self, UB: Optional[np.ndarray] = None) -> np.ndarray:
        """Return reference vector coordinates from (3, 1) NumPy array.

        Return reference vector coordinates in reciprocal of laboratory
        coordinate system. UB matrix is required when converting between
        alternative coordinate systems.

        Parameters
        ----------
        UB: np.ndarray, optional
            UB matrix as (3, 3) NumPy array object. Use UB matrix to convert
            coordinates between reciprocal and laboratory coordinate systems
            (default: None, do not convert coordinates into the alternative system)

        Returns
        -------
        np.ndarray
            Returns reference vector coordinates as (3, 1) NumPy array.

        Raises
        ------
        DiffcalcException
            If UB matrix is not a (3, 3) NumPy array.
        """
        n_ref_array = np.array([self.n_ref]).T
        if UB is None:
            return n_ref_array
        if not isinstance(UB, np.ndarray):
            raise DiffcalcException(
                "Invalid input parameter. UB must be a NumPy array object"
            )
        elif UB.shape != (3, 3):
            raise DiffcalcException(
                "Invalid array shape. UB array must have (3, 3) dimensions."
            )
        if self.rlv:
            n_ref_new = UB @ n_ref_array
        else:
            n_ref_new = inv(UB) @ n_ref_array
        return n_ref_new / norm(n_ref_new)

    def set_array(self, n_ref: np.ndarray) -> None:
        """Set reference vector coordinates from NumPy array.

        Parameters
        ----------
        n_ref: np.ndarray
            NumPy (3, 1) array with reference vector coordinates.

        Returns
        -------
        None

        Raises
        ------
        DiffcalcException
            If input parameter is not (3, 1) NumPy array

        """
        if not isinstance(n_ref, np.ndarray):
            raise DiffcalcException(
                "Input parameter in set_array method must be a NumPy array object."
            )
        if n_ref.shape != (3, 1):
            raise DiffcalcException(
                "Input NumPy array in set_array method must have (3, 1) shape."
            )
        (r1, r2, r3) = tuple(n_ref.T[0].tolist())
        self.n_ref = (r1, r2, r3)


class UBCalculation:
    """Class containing information required for for UB matrix calculation.

    Attributes
    ----------
    name: str, defalut = UUID
        Name for UB calculation. Default is to generate UUID value.
    crystal: Crystal
        Object containing crystal lattice parameters.
    reflist: ReflectionList
        List of reference reflections for UB matrix calculation.
    orientlist: OrientationList
        List of crystal orientations for UB matrix calculation.
    reference: ReferenceVector
        Object representing azimuthal reference vector.
    surface: ReferenceVector
        Object representing surface normal vector.
    U: np.ndarray
        U matrix as a NumPy array
    UB: np.ndarray
        UB matrix as a NumPy array
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name: str = name if name is not None else str(uuid.uuid4())
        self.crystal: Crystal = None
        self.reflist: ReflectionList = ReflectionList()
        self.orientlist: OrientationList = OrientationList()
        self.reference: ReferenceVector = ReferenceVector((1, 0, 0), True)
        self.surface: ReferenceVector = ReferenceVector((0, 0, 1), False)
        self.U: np.ndarray = None
        self.UB: np.ndarray = None

        self._WIDTH: int = 13

    ### State ###
    @staticmethod
    def load(filename: str) -> "UBCalculation":
        """Load current UB matrix calculation from a pickle file.

        Parameters
        ----------
        filename : str
            Pickle file name.

        Returns
        -------
        UBCalculation
            UB calculation object.

        Raises
        ------
        DiffcalcException
            If object in pickle file isn't UBcalculation class instance.
        """
        with open(filename, "rb") as fp:
            obj = pickle.load(fp)
            if isinstance(obj, UBCalculation):
                return obj
            else:
                raise DiffcalcException(
                    f"Invalid object type {type(obj)} found in pickle file."
                )

    def pickle(self, filename: str) -> None:
        """Save current UB matrix calculation as a pickle file.

        Parameters
        ----------
        filename : str
            Pickle file name.
        """
        with open(filename, "wb") as fp:
            pickle.dump(obj=self, file=fp)

    def __str__(self) -> str:
        """Text representation of UB calculation object.

        Returns
        -------
        str
            String containing crystal lattice, reference vector, UB matrix
            and reference reflection/orientation information.
        """
        if self.name is None:
            return "<<< No UB calculation started >>>"
        lines = []
        lines.append("UBCALC")
        lines.append("")
        lines.append("   name:".ljust(self._WIDTH) + self.name.rjust(9))

        lines.append("")
        lines.append("REFERNCE")
        lines.append("")
        lines.extend(
            self.__str_lines_reference(self.n_hkl, self.n_phi, self.reference.rlv)
        )
        lines.append("")
        lines.append("SURFACE NORMAL")
        lines.append("")
        lines.extend(
            self.__str_lines_reference(self.surf_nhkl, self.surf_nphi, self.surface.rlv)
        )

        lines.append("")
        lines.append("CRYSTAL")
        lines.append("")

        if self.crystal is None:
            lines.append("   <<< none specified >>>")
        else:
            lines.append(str(self.crystal))

        lines.append("")
        lines.append("UB MATRIX")
        lines.append("")

        if self.UB is None:
            lines.append("   <<< none calculated >>>")
        else:
            lines.extend(self.__str_lines_u())
            lines.append("")
            lines.extend(self.__str_lines_ub_angle_and_axis())
            lines.append("")
            lines.extend(self.__str_lines_ub())

        lines.extend(self.__str_lines_refl())

        lines.extend(self.__str_lines_orient())

        return "\n".join(lines)

    def __str_lines_u(self) -> List[str]:
        lines: List[str] = []
        if self.U is None:
            return lines
        fmt = "% 9.5f % 9.5f % 9.5f"
        lines.append(
            "   U matrix:".ljust(self._WIDTH)
            + fmt
            % (
                zero_round(self.U[0, 0]),
                zero_round(self.U[0, 1]),
                zero_round(self.U[0, 2]),
            )
        )
        lines.append(
            " " * self._WIDTH
            + fmt
            % (
                zero_round(self.U[1, 0]),
                zero_round(self.U[1, 1]),
                zero_round(self.U[1, 2]),
            )
        )
        lines.append(
            " " * self._WIDTH
            + fmt
            % (
                zero_round(self.U[2, 0]),
                zero_round(self.U[2, 1]),
                zero_round(self.U[2, 2]),
            )
        )
        return lines

    def __str_lines_ub_angle_and_axis(self) -> List[str]:
        lines = []
        fmt = "% 9.5f % 9.5f % 9.5f"
        rotation_angle, rotation_axis = self.get_miscut()
        if abs(norm(rotation_axis)) < SMALL:
            lines.append("   miscut angle:".ljust(self._WIDTH) + "  0")
        else:
            lines.append("   miscut:")
            lines.append(
                "      angle:".ljust(self._WIDTH) + "% 9.5f" % (rotation_angle * TODEG)
            )
            lines.append(
                "       axis:".ljust(self._WIDTH) + fmt % tuple(rotation_axis.tolist())
            )

        return lines

    def __str_lines_ub(self) -> List[str]:
        lines = []
        fmt = "% 9.5f % 9.5f % 9.5f"
        UB = self.UB
        lines.append(
            "   UB matrix:".ljust(self._WIDTH)
            + fmt % (zero_round(UB[0, 0]), zero_round(UB[0, 1]), zero_round(UB[0, 2]))
        )
        lines.append(
            " " * self._WIDTH
            + fmt % (zero_round(UB[1, 0]), zero_round(UB[1, 1]), zero_round(UB[1, 2]))
        )
        lines.append(
            " " * self._WIDTH
            + fmt % (zero_round(UB[2, 0]), zero_round(UB[2, 1]), zero_round(UB[2, 2]))
        )
        return lines

    def __str_vector(self, m: np.ndarray) -> str:
        if not isinstance(m, np.ndarray):
            return str(m)
        return " ".join([(f"{e:9.5f}").rjust(9) for e in m.T.tolist()[0]])

    def __str_lines_reference(
        self, ref_nhkl: np.ndarray, ref_nphi: np.ndarray, rlv: bool
    ) -> List[str]:
        SET_LABEL = " <- set"
        lines = []
        if rlv:
            nhkl_label = SET_LABEL
            nphi_label = ""
        else:
            nhkl_label = ""
            nphi_label = SET_LABEL

        lines.append(
            "   n_hkl:".ljust(self._WIDTH) + self.__str_vector(ref_nhkl) + nhkl_label
        )
        lines.append(
            "   n_phi:".ljust(self._WIDTH) + self.__str_vector(ref_nphi) + nphi_label
        )
        return lines

    def __str_lines_refl(self) -> List[str]:
        lines = ["", "REFLECTIONS", ""]

        lines.extend(self.reflist._str_lines())
        return lines

    def __str_lines_orient(self) -> List[str]:
        lines = ["", "CRYSTAL ORIENTATIONS", ""]

        lines.extend(self.orientlist._str_lines())
        return lines

    ### Lattice ###

    def set_lattice(
        self,
        name: str,
        system: Optional[
            Union[str, float]
        ] = None,  # FIXME: Cannot set Union type for  positional arguments
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        """Set crystal lattice parameters using shortform notation.

        Following combinations of system and lattice parameters are supported:

        ('Cubic', a) -- sets Cubic system
        ('Tetragonal', a, c) -- sets Tetragonal system
        ('Hexagonal', a, c) -- sets Hexagonal system
        ('Orthorhombic', a, b, c) -- sets Orthorombic system
        ('Rhombohedral', a, alpha) -- sets Rhombohedral system
        ('Monoclinic', a, b, c, beta) -- sets Monoclinic system
        ('Triclinic', a, b, c, alpha, beta, gamma) -- sets Triclinic system

        Crystal system can be inferred from the lattice parameters for the
        following cases:

        (a,) -- assumes Cubic system
        (a, c) -- assumes Tetragonal system
        (a, b, c) -- assumes Orthorombic system
        (a, b, c, angle) -- assumes Monoclinic system with beta not equal to 90 or
                            Hexagonal system if a = b and gamma = 120
        (a, b, c, alpha, beta, gamma) -- sets Triclinic system

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
        if not isinstance(name, str):
            raise TypeError("Invalid crystal name.")
        shortform = tuple(
            val for val in (system, a, b, c, alpha, beta, gamma) if val is not None
        )
        if not shortform:
            raise TypeError("Please specify unit cell parameters.")
        elif allnum(shortform):
            sf = shortform
            if len(sf) == 1:
                system = "Cubic"
            elif len(sf) == 2:
                system = "Tetragonal"
            elif len(sf) == 3:
                system = "Orthorhombic"
            elif len(sf) == 4:
                if is_small(float(sf[0]) - float(sf[1])) and sf[3] == 120:
                    system = "Hexagonal"
                else:
                    system = "Monoclinic"
            elif len(sf) == 6:
                system = "Triclinic"
            else:
                raise TypeError(
                    "Invalid number of input parameters to set unit lattice."
                )
            fullform = (system,) + shortform
        else:
            if not isinstance(shortform[0], str):
                raise TypeError("Invalid unit cell parameters specified.")
            fullform = shortform
        if self.name is None:
            raise DiffcalcException(
                "Cannot set lattice until a UBCalcaluation has been started "
                "with newubcalc"
            )
        self.crystal = Crystal(name, *fullform)

    ### Reference vector ###
    @property
    def n_hkl(self) -> np.ndarray:
        """Return reference vector property represented using miller indices.

        Returns
        -------
        np.ndarray:
            Reference vector represented as (3,1) NumPy array.
        """
        if self.UB is None and not self.reference.rlv:
            return None
        return self.reference.get_array(None if self.reference.rlv else self.UB)

    @n_hkl.setter
    def n_hkl(self, n_hkl: Tuple[float, float, float]) -> None:
        self.reference = ReferenceVector(deepcopy(n_hkl), True)

    @property
    def n_phi(self) -> np.ndarray:
        """Return reference vector property represented laboratory space coordinates.

        Returns
        -------
        np.ndarray:
            Reference vector represented as (3,1) NumPy array.
        """
        if self.UB is None and self.reference.rlv:
            return None
        return self.reference.get_array(self.UB if self.reference.rlv else None)

    @n_phi.setter
    def n_phi(self, n_phi: Tuple[float, float, float]) -> None:
        self.reference = ReferenceVector(deepcopy(n_phi), False)

    ### Surface vector ###
    @property
    def surf_nhkl(self) -> np.ndarray:
        """Return surface normal vector property represented using miller indices.

        Returns
        -------
        np.ndarray:
            Surface normal vector represented as (3,1) NumPy array.
        """
        return self.surface.get_array(None if self.surface.rlv else self.UB)

    @surf_nhkl.setter
    def surf_nhkl(self, surf_nhkl: Tuple[float, float, float]) -> None:
        self.surface = ReferenceVector(deepcopy(surf_nhkl), True)

    @property
    def surf_nphi(self) -> np.ndarray:
        """Return surface normal vector represented using laboratory space coordinates.

        Returns
        -------
        np.ndarray:
            Reference vector represented as (3,1) NumPy array.
        """
        return self.surface.get_array(self.UB if self.surface.rlv else None)

    @surf_nphi.setter
    def surf_nphi(self, surf_nphi: Tuple[float, float, float]) -> None:
        self.surface = ReferenceVector(deepcopy(surf_nphi), False)

    ### Reflections ###

    def add_reflection(
        self,
        hkl: Tuple[float, float, float],
        position: Position,
        energy: float,
        tag: Optional[str] = None,
    ):
        """Add a reference reflection.

        Adds a reflection position in degrees and in the systems internal
        representation.

        Parameters
        ----------
        hkl : Tuple[float, float, float]
            hkl index of the reflection
        position: Position
            list of diffractometer angles in internal representation in degrees
        energy : float
            energy of the x-ray beam
        tag : Optional[str], default = None
            identifying tag for the reflection
        """
        if self.reflist is None:
            raise DiffcalcException("No UBCalculation loaded")
        self.reflist.add_reflection(hkl, position, energy, tag)

    def edit_reflection(
        self,
        idx: Union[str, int],
        hkl: Tuple[float, float, float],
        position: Position,
        energy: float,
        tag: Optional[str] = None,
    ):
        """Change a reference reflection.

        Changes a reflection position in degrees and in the
        systems internal representation.

        Parameters
        ----------
        idx : Union[str, int]
            index or tag of the reflection to be changed
        hkl : Tuple[float, float, float]
            hkl index of the reflection
        position: Position
            list of diffractometer angles in internal representation in degrees
        energy : float
            energy of the x-ray beam
        tag : Optional[str], default = None
            identifying tag for the reflection
        """
        if self.reflist is None:
            raise DiffcalcException("No UBCalculation loaded")
        self.reflist.edit_reflection(idx, hkl, position, energy, tag)

    def get_reflection(self, idx: Union[str, int]) -> Reflection:
        """Get a reference reflection.

        Get a reflection position in degrees and in the
        systems internal representation.

        Parameters
        ----------
        idx : Union[str, int]
            index or tag of the reflection
        """
        return self.reflist.get_reflection(idx)

    def get_number_reflections(self) -> int:
        """Get a number of stored reference reflections.

        Returns
        -------
        int:
            Number of reference reflections.
        """
        try:
            return len(self.reflist)
        except TypeError:
            return 0

    def get_tag_refl_num(self, tag: str) -> int:
        """Get a reference reflection index.

        Get a reference reflection index for the
        provided reflection tag.

        Parameters
        ----------
        tag : str
            identifying tag for the reflection
        Returns
        -------
        int:
            Reference reflection index
        """
        if tag is None:
            raise IndexError("Reflection tag is None")
        return self.reflist.get_tag_index(tag) + 1

    def del_reflection(self, idx: Union[str, int]) -> None:
        """Delete a reference reflection.

        Parameters
        ----------
        idx : str or int
            index or tag of the deleted reflection
        """
        self.reflist.remove_reflection(idx)

    def swap_reflections(self, idx1: Union[str, int], idx2: Union[str, int]) -> None:
        """Swap indices of two reference reflections.

        Parameters
        ----------
        idx1 : str or int
            index or tag of the first reflection to be swapped
        idx2 : str or int
            index or tag of the second reflection to be swapped
        """
        self.reflist.swap_reflections(idx1, idx2)

    ### Orientations ###

    def add_orientation(self, hkl, xyz, position=None, tag=None):
        """Add a reference orientation.

        Adds a reference orientation in the diffractometer
        coordinate system.

        Parameters
        ----------
        hkl : :obj:`tuple` of numbers
            hkl index of the reference orientation
        xyz : :obj:`tuple` of numbers
            xyz coordinate of the reference orientation
        position: :obj:`list` or :obj:`tuple` of numbers
            list of diffractometer angles in internal representation in degrees
        tag : str
            identifying tag for the reflection
        """
        if self.orientlist is None:
            raise DiffcalcException("No UBCalculation loaded")
        if position is None:
            position = Position()
        self.orientlist.add_orientation(hkl, xyz, position, tag)

    def edit_orientation(self, idx, hkl, xyz, position=None, tag=None):
        """Change a reference orientation.

        Changes a reference orientation in the diffractometer
        coordinate system.

        Parameters
        ----------
        idx : str or int
            index or tag of the orientation to be changed
        hkl : :obj:`tuple` of numbers
            h index of the reference orientation
        xyz : :obj:`tuple` of numbers
            x coordinate of the reference orientation
        position: :obj:`list` or :obj:`tuple` of numbers
            list of diffractometer angles in internal representation in degrees
        tag : str
            identifying tag for the reflection
        """
        if self.orientlist is None:
            raise DiffcalcException("No UBCalculation loaded")
        self.orientlist.edit_orientation(
            idx, hkl, xyz, position if position is not None else Position(), tag
        )

    def get_orientation(self, idx):
        """Get a reference orientation.

        Get a reference orientation in the diffractometer
        coordinate system.

        Parameters
        ----------
        idx : str or int
            index or tag of the reference orientation
        """
        return self.orientlist.get_orientation(idx)

    def get_number_orientations(self):
        """Get a number of stored reference orientations.

        Returns
        -------
        int:
            Number of reference orientations.
        """
        try:
            return len(self.orientlist)
        except TypeError:
            return 0

    def get_tag_orient_num(self, tag):
        """Get a reference orientation index.

        Get a reference orientation index for the
        provided orientation tag.

        Parameters
        ----------
        tag : str
            identifying tag for the orientation
        Returns
        -------
        int:
            Reference orientation index
        """
        if tag is None:
            raise IndexError("Orientations tag is None")
        return self.orientlist.get_tag_index(tag) + 1

    def del_orientation(self, idx):
        """Delete a reference orientation.

        Parameters
        ----------
        idx : str or int
            index or tag of the deleted orientation
        """
        self.orientlist.remove_orientation(idx)

    def swap_orientations(self, idx1, idx2):
        """Swap indices of two reference orientations.

        Parameters
        ----------
        idx1 : str or int
            index or tag of the first orientation to be swapped
        idx2 : str or int
            index or tag of the second orientation to be swapped
        """
        self.orientlist.swap_orientations(idx1, idx2)

    ### Calculations ###

    def set_u(
        self,
        matrix: Union[
            np.ndarray,
            List[List[float]],
            Tuple[
                Tuple[float, float, float],
                Tuple[float, float, float],
                Tuple[float, float, float],
            ],
        ],
    ) -> None:
        """Manually sets U matrix.

        Set U matrix from input values. Update UB matrix if crystal lattice
        information is set.

        Parameters
        ----------

        matrix: Union(np.ndarray,
                      List[List[float]],
                      Tuple[Tuple[float, float, float],
                            Tuple[float, float, float],
                            Tuple[float, float, float]]
            Collection containing U matrix coordinates.

        Raises
        ------
        ValueError
            if collection isn't (3,3) shape.
        """
        if type(matrix) in (list, tuple):
            m = np.array(matrix, dtype="object")
        else:
            m = np.array(matrix)
        if not isinstance(m, np.ndarray):
            raise TypeError("Invalid input U matrix type")
        if len(m.shape) != 2 or m.shape[0] != 3 or m.shape[1] != 3:
            raise TypeError("Expects (3, 3) NumPy matrix.")

        if self.UB is None:
            print("Calculating UB matrix.")
        else:
            print("Recalculating UB matrix.")

        self.U = m
        if self.crystal is not None:
            self.UB = self.U @ self.crystal.B

    def set_ub(self, m):
        """Manually sets U matrix.

        Parameters
        ----------

        matrix: Union(np.ndarray,
                      List[List[float]],
                      Tuple[Tuple[float, float, float],
                            Tuple[float, float, float],
                            Tuple[float, float, float]]
            Collection containing U matrix coordinates.

        Raises
        ------
        ValueError
            if collection isn't (3,3) shape.
        """
        """Manually sets UB. matrix must be 3*3 Jama or python matrix.
        Turns off aution UB calcualtion.
        """
        if type(m) in (list, tuple):
            m = np.array(m, dtype="object")
        if not isinstance(m, np.ndarray):
            raise TypeError("Invalid input UB matrix type")
        if len(m.shape) != 2 or m.shape[0] != 3 or m.shape[1] != 3:
            raise TypeError("Expects 3*3 matrix")

        self.UB = m

    def _calc_ub_from_two_references(self, ref1, ref2):
        h1 = np.array([[ref1.h, ref1.k, ref1.l]]).T  # row->column
        h2 = np.array([[ref2.h, ref2.k, ref2.l]]).T

        # Compute the two reflections' reciprocal lattice vectors in the
        # cartesian crystal frame
        up = []
        for ref in (ref1, ref2):
            try:
                [MU, _, _, ETA, CHI, PHI] = get_rotation_matrices(ref.pos)
                Z = inv(PHI) @ inv(CHI) @ inv(ETA) @ inv(MU)
                up.append(Z @ np.array([[ref.x, ref.y, ref.z]]).T)
            except AttributeError:
                up.append(get_q_phi(ref.pos))
        u1p, u2p = up

        B = self.crystal.B
        h1c = B @ h1
        h2c = B @ h2

        # Create modified unit vectors t1, t2 and t3 in crystal and phi systems
        t1c = h1c
        t3c = cross3(h1c, h2c)
        t2c = cross3(t3c, t1c)

        t1p = u1p  # FIXED from h1c 9July08
        t3p = cross3(u1p, u2p)
        t2p = cross3(t3p, t1p)

        # ...and nornmalise and check that the reflections used are appropriate

        def __normalise(m):
            SMALL = 1e-4  # Taken from Vlieg's code
            d = norm(m)
            if d < SMALL:
                raise DiffcalcException(
                    "Invalid UB reference data. Please check that the specified "
                    "reference reflections/orientations are not parallel."
                )
            return m / d

        t1c = __normalise(t1c)
        t2c = __normalise(t2c)
        t3c = __normalise(t3c)

        t1p = __normalise(t1p)
        t2p = __normalise(t2p)
        t3p = __normalise(t3p)

        Tc = np.hstack([t1c, t2c, t3c])
        Tp = np.hstack([t1p, t2p, t3p])
        self.U = Tp @ inv(Tc)
        self.UB = self.U @ B

    def _get_calc_ub_references(self, idx1=None, idx2=None):
        try:
            ref1 = self.get_reflection(idx1)
        except Exception:
            try:
                ref1 = self.get_orientation(idx1)
            except Exception:
                raise DiffcalcException(
                    "Cannot find first reflection or orientation with index %s"
                    % str(idx1)
                )
        try:
            ref2 = self.get_reflection(idx2)
        except Exception:
            try:
                ref2 = self.get_orientation(idx2)
            except Exception:
                raise DiffcalcException(
                    "Cannot find second reflection or orientation with index %s"
                    % str(idx2)
                )
        return ref1, ref2

    def calc_ub(self, idx1=None, idx2=None):
        """Calculate UB matrix.

        Calculate UB matrix using two reference reflections and/or
        reference orientations.

        By default use the first two reference reflections when provided.
        If one or both reflections are not available use one or two reference
        orientations to complement mission reflection data.

        Parameters
        ----------
        idx1: int or str, optional
            The index or the tag of the first reflection or orientation.
        idx2: int or str, optional
            The index or the tag of the second reflection or orientation.
        """
        # Major variables:
        # h1, h2: user input reciprocal lattice vectors of the two reflections
        # h1c, h2c: user input vectors in cartesian crystal plane
        # pos1, pos2: measured diffractometer positions of the two reflections
        # u1a, u2a: measured reflection vectors in alpha frame
        # u1p, u2p: measured reflection vectors in phi frame

        # Get hkl and angle values for the first two reflections
        if self.crystal is None:
            raise DiffcalcException(
                "Not calculating UB matrix as no lattice parameters have "
                "been specified."
            )

        if idx1 is not None and idx2 is None:
            self._calc_ub_from_primary_only(idx1)
            return
        elif idx1 is None and idx2 is None:
            if self.get_number_reflections() == 1:
                self._calc_ub_from_primary_only()
                return
            ref_data = []
            for func, idx in product(
                (self.get_reflection, self.get_orientation), (1, 2)
            ):
                try:
                    ref_data.append(func(idx))
                except Exception:
                    pass
            try:
                ref1, ref2 = ref_data[:2]
            except ValueError:
                raise DiffcalcException(
                    "Cannot find reference vectors to calculate a U matrix. "
                    "Please add reference reflection and/or orientation data."
                )
        else:
            ref1, ref2 = self._get_calc_ub_references(idx1, idx2)

        self._calc_ub_from_two_references(ref1, ref2)

    def _calc_ub_from_primary_only(self, idx: int = 1) -> None:
        """Calculate orientation matrix from a single reference reflection.

        Parameters
        ----------
        idx: int
            Reference reflection index, default is 1.
        """
        # Algorithm from http://www.j3d.org/matrix_faq/matrfaq_latest.html

        # Get hkl and angle values for the first two reflections
        if self.crystal is None:
            raise DiffcalcException(
                "Not calculating UB matrix as no lattice parameters have "
                "been specified."
            )

        try:
            refl = self.get_reflection(idx)
        except IndexError:
            raise DiffcalcException(
                "One reflection is required to calculate a u matrix"
            )

        h = np.array([[refl.h], [refl.k], [refl.l]])
        B = self.crystal.B
        h_crystal = B @ h
        h_crystal = h_crystal * (1 / norm(h_crystal))

        q_measured_phi = get_q_phi(refl.pos)
        q_measured_phi = q_measured_phi * (1 / norm(q_measured_phi))

        rotation_axis = cross3(h_crystal, q_measured_phi)
        rotation_axis = rotation_axis * (1 / norm(rotation_axis))

        cos_rotation_angle = dot3(h_crystal, q_measured_phi)
        rotation_angle = acos(cos_rotation_angle)

        uvw = rotation_axis.T.tolist()[0]  # TODO: cleanup
        print("resulting U angle: %.5f deg" % (rotation_angle * TODEG))
        u_repr = ", ".join(["% .5f" % el for el in uvw])
        print("resulting U axis direction: [%s]" % u_repr)

        u, v, w = uvw
        rcos = cos(rotation_angle)
        rsin = sin(rotation_angle)
        m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # TODO: tidy
        m[0][0] = rcos + u * u * (1 - rcos)
        m[1][0] = w * rsin + v * u * (1 - rcos)
        m[2][0] = -v * rsin + w * u * (1 - rcos)
        m[0][1] = -w * rsin + u * v * (1 - rcos)
        m[1][1] = rcos + v * v * (1 - rcos)
        m[2][1] = u * rsin + w * v * (1 - rcos)
        m[0][2] = v * rsin + u * w * (1 - rcos)
        m[1][2] = -u * rsin + v * w * (1 - rcos)
        m[2][2] = rcos + w * w * (1 - rcos)

        if self.UB is None:
            print("Calculating UB matrix from the first reflection only.")
        else:
            print("Recalculating UB matrix from the first reflection only.")
        print(
            "NOTE: A new UB matrix will not be automatically calculated "
            "when the orientation reflections are modified."
        )

        self.U = np.array(m)
        self.UB = self.U @ B

    def refine_ub(
        self,
        hkl: Tuple[float, float, float],
        position: Position,
        wavelength: float,
        refine_lattice=False,
        refine_umatrix=False,
    ):
        """Refine UB matrix to using single reflection.

        Refine UB matrix to match diffractometer position for the specified
        reflection. Refined U matrix will be accurate up to an azimuthal rotation
        around the specified scattering vector.

        Parameters
        ----------
        hkl: Tuple[float, float, float]
            Miller indices of the reflection.
        pos: Position
            Diffractometer position object.
        wavelength: float
            Radiation wavelength.
        refine_lattice: Optional[bool], default = False
            Apply refined lattice parameters to the current UB calculation object.
        refine_umatrix: Optional[bool], default = False
            Apply refined U matrix to the current UB calculation object.

        Returns
        -------
        Tuple[np.ndarray, Tuple[str, float, float, float, float, float, float]]
            Refined U matrix as NumPy array and refined crystal lattice parameters.
        """
        """
        refineub {[h k l]} {pos} -- refine unit cell dimensions and U matrix to match diffractometer angles for a given hkl value
        """
        scale, lat = self._rescale_unit_cell(hkl, position, wavelength)
        if scale and refine_lattice:
            self.set_lattice(*lat)
        mc_angle, mc_axis = self.get_miscut_from_hkl(hkl, position)
        if mc_angle and refine_umatrix:
            self.set_miscut(mc_axis, mc_angle * TORAD, True)

    def fit_ub(
        self,
        indices: Sequence[Union[str, int]],
        refine_lattice: Optional[bool] = False,
        refine_umatrix: Optional[bool] = False,
    ) -> Tuple[np.ndarray, Tuple[str, float, float, float, float, float, float]]:
        """Refine UB matrix using reference reflections.

        Parameters
        ----------
        indices: Sequence[Union[str, int]]
            List of reference reflection indices or tags.
        refine_lattice: Optional[bool], default = False
            Apply refined lattice parameters to the current UB calculation object.
        refine_umatrix: Optional[bool], default = False
            Apply refined U matrix to the current UB calculation object.

        Returns
        -------
        Tuple[np.ndarray, Tuple[str, float, float, float, float, float, float]]
            Refined U matrix as NumPy array and refined crystal lattice parameters.
        """
        if indices is None:
            raise DiffcalcException(
                "Please specify list of reference reflection indices."
            )
        if len(indices) < 3:
            raise DiffcalcException(
                "Need at least 3 reference reflections to fit UB matrix."
            )

        if len(self.crystal.get_lattice_params()[1]) == 6:
            new_u, new_lattice = self._fit_ub_uncon(indices)
        else:
            refl_list = []
            for idx in indices:
                try:
                    refl = self.get_reflection(idx)
                    refl_list.append(refl)
                except IndexError:
                    raise DiffcalcException(
                        "Cannot read reflection data for index %s" % str(idx)
                    )
            print("Fitting crystal lattice parameters...")
            new_crystal = fit_crystal(self.crystal, refl_list)
            print("Fitting orientation matrix...")
            new_u = fit_u_matrix(self.U, new_crystal, refl_list)
            new_lattice = (self.crystal.get_lattice()[0],) + new_crystal.get_lattice()[
                1:
            ]

        crystal_system = self.crystal.system

        if refine_lattice:
            self.set_lattice(new_lattice[0], crystal_system, *new_lattice[1:])

        if refine_umatrix:
            self.set_u(new_u)
        return new_u, new_lattice

    def _fit_ub_uncon(
        self, indices: Sequence[Union[str, int]]
    ) -> Tuple[np.ndarray, Tuple[str, float, float, float, float, float, float]]:
        """Refine UB matrix using least-squares solution."""
        if indices is None:
            raise DiffcalcException(
                "Please specify list of reference reflection indices."
            )
        if len(indices) < 3:
            raise DiffcalcException(
                "Need at least 3 reference reflections to fit UB matrix."
            )

        x = []
        y = []
        for idx in indices:
            try:
                refl = self.get_reflection(idx)
            except IndexError:
                raise DiffcalcException(
                    "Cannot read reflection data for index %s" % str(idx)
                )
            x.append((refl.h, refl.k, refl.l))
            wl = 12.3984 / refl.energy
            y_tmp = get_q_phi(refl.pos) * 2.0 * pi / wl
            y.append(y_tmp.T.tolist()[0])

        xm = np.array(x)
        ym = np.array(y)
        b = inv(xm.T @ xm) @ xm.T @ ym

        b1, b2, b3 = np.array([b[0]]), np.array([b[1]]), np.array([b[2]])
        e1 = b1 / norm(b1)
        e2 = b2 - e1 * dot3(b2.T, e1.T)
        e2 = e2 / norm(e2)
        e3 = b3 - e1 * dot3(b3.T, e1.T) - e2 * dot3(b3.T, e2.T)
        e3 = e3 / norm(e3)

        new_umatrix = np.array(e1.tolist() + e2.tolist() + e3.tolist()).T

        V = dot3(cross3(b1.T, b2.T), b3.T)
        a1 = cross3(b2.T, b3.T) * 2 * pi / V
        a2 = cross3(b3.T, b1.T) * 2 * pi / V
        a3 = cross3(b1.T, b2.T) * 2 * pi / V
        ax, bx, cx = norm(a1), norm(a2), norm(a3)
        alpha = acos(dot3(a2, a3) / (bx * cx)) * TODEG
        beta = acos(dot3(a1, a3) / (ax * cx)) * TODEG
        gamma = acos(dot3(a1, a2) / (ax * bx)) * TODEG

        lattice_name = self.crystal.get_lattice()[0]
        return new_umatrix, (lattice_name, ax, bx, cx, alpha, beta, gamma)

    def get_miscut(self) -> Tuple[float, np.ndarray]:
        """Calculate miscut angle and axis from U matrix.

        Returns
        -------
        Tuple[float, np.ndarray]
            Miscut angle and miscut axis as (3,1) NumPy array.
        """
        rotvec = Rotation.from_matrix(self.U).as_rotvec()
        rot_angle = norm(rotvec)
        if is_small(rot_angle):
            return 0, np.array([0, 0, 0])
        rot_axis = rotvec / rot_angle
        # surf_rot = self.UB @ self.surf_nphi
        # rotation_axis = cross3(self.surf_nphi, surf_rot)
        # if abs(norm(rotation_axis)) < SMALL:
        #    rotation_axis = array([[0], [0], [0]])
        #    rotation_angle = 0
        # else:
        #    rotation_axis = rotation_axis * (1 / norm(rotation_axis))
        #    cos_rotation_angle = bound(dot3(self.surf_nphi, surf_rot) / norm(surf_rot))
        #    rotation_angle = acos(cos_rotation_angle) * TODEG
        return rot_angle, rot_axis

    def get_miscut_from_hkl(
        self, hkl: Tuple[float, float, float], pos: Position
    ) -> Tuple[float, Tuple[float, float, float]]:
        """Calculate miscut angle and axis from a single reflection.

        Using single reflection U matrix can be accurate up to an azimuthal
        rotation around the specified scattering vector.

        Parameters
        ----------
        hkl: Tuple[float, float, float]
            Miller indices of the reflection.
        pos: Position
            Diffractometer position object.

        Returns
        -------
        Tuple[float, Tuple[float, float, float]]
            The miscut angle and the corresponding miscut rotation axis.
        """
        q_vec = get_q_phi(pos)
        hkl_nphi = self.UB @ np.array([hkl]).T
        axis = cross3(hkl_nphi, q_vec)
        norm_axis = norm(axis)
        if norm_axis < SMALL:
            return None, None
        axis = axis / norm(axis)
        try:
            miscut = (
                acos(bound(dot3(q_vec, hkl_nphi) / (norm(q_vec) * norm(hkl_nphi))))
                * TODEG
            )
        except AssertionError:
            return 0, (0, 0, 0)
        return miscut, (axis[0, 0], axis[1, 0], axis[2, 0])

    def set_miscut(
        self,
        xyz: Tuple[float, float, float],
        angle: float,
        add_miscut: Optional[bool] = False,
    ) -> None:
        """Calculate U matrix using a miscut axis and an angle.

        Parameters
        ----------
        xyz: Tuple[float, float, float]
            Rotation axis corresponding to the crystal miscut.
        angle: float
            The miscut angle.
        add_miscut: Optional[bool], default = False
            If False, set crystal miscut to the provided parameters.
            If True, apply provided miscut parameters to the existing settings.
        """
        if xyz is None:
            rot_matrix = xyz_rotation((0, 1, 0), angle)
        else:
            rot_matrix = xyz_rotation(xyz, angle)
        if self.U is not None and add_miscut:
            new_U = rot_matrix @ self.U
        else:
            new_U = rot_matrix
        self.set_u(new_U)

    def get_ttheta_from_hkl(self, hkl: Tuple[float, float, float], en: float) -> float:
        """Calculate two-theta scattering angle for a reflection.

        Parameters
        ----------
        hkl:  Tuple[float, float, float]
            Miller indices of the reflection.
        en: float
            Beam energy.

        Returns
        -------
        float
            Two-theta angle for the reflection.

        Raises
        ------
        ValueError
            If reflection is unreachable at the provided energy.
        """
        if self.crystal is None:
            raise DiffcalcException(
                "Cannot calculate theta angle as no lattice parameters have been specified."
            )
        wl = 12.39842 / en
        d = self.crystal.get_hkl_plane_distance(hkl)
        if wl > (2 * d):
            raise ValueError(
                "Reflection un-reachable as wavelength (%f) is more than twice\n"
                "the plane distance (%f)" % (wl, d)
            )
        try:
            return 2.0 * asin(wl / (d * 2))
        except ValueError as e:
            raise ValueError(
                f"asin(wl / (d * 2) with wl={wl:f} and d={d:f}: " + e.args[0]
            )

    def _rescale_unit_cell(
        self, hkl: Tuple[float, float, float], pos: Position, wavelength: float
    ) -> Tuple[float, Tuple[str, str, float, float, float, float, float, float]]:
        """Calculate unit cell scaling factor.

        Match lattice spacing corresponding to a given miller indices to the
        provided diffractometer position.

        Parameters
        ----------
        hkl: Tuple[float, float, float]
            Reflection mille indices.
        pos: Position
            Diffractometer position object.

        Returns
        -------
        Tuple[float, Tuple[str, str, float, float, float, float, float, float]]
            Scaling factor and updated crystal lattice parameters.
        """
        q_vec = get_q_phi(pos)
        q_hkl = norm(q_vec) / wavelength
        d_hkl = self.crystal.get_hkl_plane_distance(hkl)
        sc = 1 / (q_hkl * d_hkl)
        name, a1, a2, a3, alpha1, alpha2, alpha3 = self.crystal.get_lattice()
        if abs(sc - 1.0) < SMALL:
            return None, None
        h, k, l = hkl
        ref_a1 = sc * a1 if abs(h) > SMALL else a1
        ref_a2 = sc * a2 if abs(k) > SMALL else a2
        ref_a3 = sc * a3 if abs(l) > SMALL else a3
        return sc, (
            name,
            self.crystal.system,
            ref_a1,
            ref_a2,
            ref_a3,
            alpha1,
            alpha2,
            alpha3,
        )
