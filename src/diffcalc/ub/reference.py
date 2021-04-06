"""Module providing objects for working with reference reflections and orientations."""
import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Union

from diffcalc.hkl.geometry import Position


@dataclass
class Reflection:
    """Class containing reference reflection information.

    Attributes
    ----------
    h: float
        h miller index.
    k: float
        k miller index.
    l: float
        l miller index.
    pos: Position
        Diffractometer position object.
    energy: float
        Beam energy in keV.
    tag: str
        Identifying tag for the reflection.
    """

    h: float
    k: float
    l: float
    pos: Position
    energy: float
    tag: str

    def __post_init__(self):
        """Check input argument types.

        Raises
        ------
        TypeError
            If pos argument has invalid type.
        """
        if not isinstance(self.pos, Position):
            raise TypeError(f"Invalid position object type {type(self.pos)}.")

    @property
    def astuple(
        self,
    ) -> Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float, float, float, float],
        float,
        str,
    ]:
        """Return reference reflection data as tuple.

        Returns
        -------
        Tuple[Tuple[float, float, float],
              Tuple[float, float, float, float, float, float],
              float,
              str]
            Tuple containing miller indices, position object, energy and
            reflection tag.
        """
        h, k, l, pos, en, tag = dataclasses.astuple(self)
        return (h, k, l), pos, en, tag


class ReflectionList:
    """Class containing collection of reference reflections.

    Attributes
    ----------
    reflections: List[Reflection]
        List containing reference reflections.
    """

    def __init__(self, reflections=None):
        self.reflections: List[Reflection] = reflections if reflections else []

    def get_tag_index(self, tag: str) -> int:
        """Get a reference reflection index.

        Get a reference reflection index for the provided reflection tag.

        Parameters
        ----------
        tag : str
            identifying tag for the reflection

        Returns
        -------
        int:
            The reference reflection index.

        Raises
        ------
        ValueError
            If tag not found in reflection list.
        """
        _tag_list = [ref.tag for ref in self.reflections]
        num = _tag_list.index(tag)
        return num

    def add_reflection(
        self, hkl: Tuple[float, float, float], pos: Position, energy: float, tag: str
    ) -> None:
        """Add a reference reflection.

        Adds a reference reflection object to the reflection list.

        Parameters
        ----------
        hkl : Tuple[float, float, float]
            Miller indices of the reflection
        pos: Position
            Object representing diffractometer angles
        energy : float
            Energy of the x-ray beam.
        tag : str
            Identifying tag for the reflection.
        """
        self.reflections += [Reflection(*hkl, pos, energy, tag)]

    def edit_reflection(
        self,
        idx: Union[str, int],
        hkl: Tuple[float, float, float],
        pos: Position,
        energy: float,
        tag: str,
    ) -> None:
        """Change a reference reflection.

        Changes the reference reflection object in the reflection list.

        Parameters
        ----------
        idx : Union[str, int]
            Index or tag of the reflection to be changed
        hkl : Tuple[float,float,float]
            Miller indices of the reflection
        position: Position
            Object representing diffractometer angles.
        energy : float
            Energy of the x-ray beam.
        tag : str
            Identifying tag for the reflection.

        Raises
        ------
        ValueError
            Reflection with specified tag not found.
        IndexError
            Reflection with specified index not found.
        """
        if isinstance(idx, str):
            num = self.get_tag_index(idx)
        else:
            num = idx - 1
        if isinstance(pos, Position):
            self.reflections[num] = Reflection(*hkl, pos, energy, tag)
        else:
            raise TypeError("Invalid position parameter type")

    def get_reflection(self, idx: Union[str, int]) -> Reflection:
        """Get a reference reflection.

        Get aon object representing reference reflection.

        Parameters
        ----------
        idx : Union[str, int]
            Index or tag of the reflection.

        Returns
        -------
        Reflection
            Object representing reference reflection

        Raises
        ------
        ValueError
            Reflection with the requested index/tan not present.
        IndexError
            Reflection with specified index not found.
        """
        if isinstance(idx, str):
            num = self.get_tag_index(idx)
        else:
            num = idx - 1
        return self.reflections[num]

    def remove_reflection(self, idx: Union[str, int]) -> None:
        """Delete a reference reflection.

        Parameters
        ----------
        idx : Union[str, int]
            Index or tag of the deleted reflection.

        Raises
        ------
        ValueError
            Reflection with the requested index/tag not present.
        IndexError
            Reflection with specified index not found.
        """
        if isinstance(idx, str):
            num = self.get_tag_index(idx)
        else:
            num = idx - 1
        del self.reflections[num]

    def swap_reflections(self, idx1: Union[str, int], idx2: Union[str, int]) -> None:
        """Swap indices of two reference reflections.

        Parameters
        ----------
        idx1 : Union[str, int]
            Index or tag of the first reflection to be swapped.
        idx2 : Union[str, int]
            Index or tag of the second reflection to be swapped.

        Raises
        ------
        ValueError
            Reflection with the requested index/tag not present.
        IndexError
            Reflection with specified index not found.
        """
        if isinstance(idx1, str):
            num1 = self.get_tag_index(idx1)
        else:
            num1 = idx1 - 1
        if isinstance(idx2, str):
            num2 = self.get_tag_index(idx2)
        else:
            num2 = idx2 - 1
        orig1 = self.reflections[num1]
        self.reflections[num1] = self.reflections[num2]
        self.reflections[num2] = orig1

    def __len__(self) -> int:
        """Return number of reference reflections in the list.

        Returns
        -------
        int
            Number of reference reflections.
        """
        return len(self.reflections)

    def __str__(self) -> str:
        """Represent the reference reflection list as a string.

        Returns
        -------
        str
            Table containing list of all reflections.
        """
        return "\n".join(self._str_lines())

    def _str_lines(self) -> List[str]:
        """Table with reference reflection data.

        Returns
        -------
        List[str]
            List containing reference reflection table rows.
        """
        axes = tuple(fd.name.upper() for fd in dataclasses.fields(Position))
        if not self.reflections:
            return ["   <<< none specified >>>"]

        lines = []

        fmt = "     %6s %5s %5s %5s  " + "%8s " * len(axes) + " TAG"
        header_values = ("ENERGY", "H", "K", "L") + axes
        lines.append(fmt % header_values)

        for n in range(1, len(self.reflections) + 1):
            ref_tuple = self.get_reflection(n)
            (h, k, l), pos, energy, tag = ref_tuple.astuple
            if tag is None:
                tag = ""
            fmt = "  %2d %6.3f % 4.2f % 4.2f % 4.2f  " + "% 8.4f " * len(axes) + " %s"
            values = (n, energy, h, k, l) + pos + (tag,)
            lines.append(fmt % values)
        return lines


@dataclass
class Orientation:
    """Class containing reference orientation information.

    Attributes
    ----------
    h: float
        h miller index.
    k: float
        k miller index.
    l: float
        l miller index.
    x: float
        x coordinate in laboratory system.
    y: float
        y coordinate in laboratory system.
    z: float
        z coordinate in laboratory system.
    pos: Position
        Diffractometer position object.
    tag: str
        Identifying tag for the orientation.
    """

    h: float
    k: float
    l: float
    x: float
    y: float
    z: float
    pos: Position
    tag: str

    def __post_init__(self):
        """Check input argument types.

        Raises
        ------
        TypeError
            If pos argument has invalid type.
        """
        if not isinstance(self.pos, Position):
            raise TypeError(f"Invalid position object type {type(self.pos)}.")

    @property
    def astuple(
        self,
    ) -> Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float, float, float, float],
        str,
    ]:
        """Return reference orientation data as tuple.

        Returns
        -------
        Tuple[Tuple[float, float, float],
              Tuple[float, float, float],
              Tuple[float, float, float, float, float, float],
              str]
            Tuple containing miller indices, laboratory frame coordinates,
            position object and orientation tag.
        """
        h, k, l, x, y, z, pos, tag = dataclasses.astuple(self)
        return (h, k, l), (x, y, z), pos, tag


class OrientationList:
    """Class containing collection of reference orientations.

    Attributes
    ----------
    reflections: List[Orientation]
        List containing reference orientations.
    """

    def __init__(self, orientations=None):
        self.orientations: List[Orientation] = orientations if orientations else []

    def get_tag_index(self, tag: str) -> int:
        """Get a reference orientation index.

        Get a reference orientation index for the provided orientation tag.

        Parameters
        ----------
        tag : str
            identifying tag for the orientation

        Returns
        -------
        int:
            The reference orientation index.

        Raises
        ------
        ValueError
            If tag not found in orientations list.
        """
        _tag_list = [orient.tag for orient in self.orientations]
        num = _tag_list.index(tag)
        return num

    def add_orientation(
        self,
        hkl: Tuple[float, float, float],
        xyz: Tuple[float, float, float],
        pos: Position,
        tag: str,
    ) -> None:
        """Add a reference orientation.

        Adds a reference orientation in the external diffractometer
        coordinate system.

        Parameters
        ----------
        hkl : Tuple[float, float, float]
            Miller index of the reference orientation.
        xyz : Tuple[float, float, float]
            Laboratory frame coordinates of the reference orientation.
        position: Position
            Object representing diffractometer position.
        tag : str
            identifying tag for the orientation.
        """
        if isinstance(pos, Position):
            self.orientations += [Orientation(*hkl, *xyz, pos, tag)]
        else:
            raise TypeError("Invalid position parameter type")

    def edit_orientation(
        self,
        idx: Union[str, int],
        hkl: Tuple[float, float, float],
        xyz: Tuple[float, float, float],
        pos: Position,
        tag: str,
    ) -> None:
        """Change a reference orientation.

        Changes a reference orientation in the external diffractometer
        coordinate system.

        Parameters
        ----------
        idx : str or int
            Index or tag of the orientation to be changed.
        hkl : Tuple[float, float, float]
            Miller indices of the reference orientation.
        xyz : :Tuple[float, float, float]
            Laboratory frame coordinates of the reference orientation.
        pos: Position
            Object representing diffractometer position.
        tag : str
            Identifying tag for the orientation.

        Raises
        ------
        ValueError
            Orientation with specified tag not found.
        IndexError
            Orientation with specified index not found.
        """
        if isinstance(idx, str):
            num = self.get_tag_index(idx)
        else:
            num = idx - 1
        if isinstance(pos, Position):
            self.orientations[num] = Orientation(*hkl, *xyz, pos, tag)
        else:
            raise TypeError(f"Invalid position parameter type {type(pos)}")

    def get_orientation(self, idx: Union[str, int]) -> Orientation:
        """Get a reference orientation.

        Get an object representing reference orientation.

        Parameters
        ----------
        idx : Union[str, int]
            Index or tag of the orientation.

        Returns
        -------
        Orientation
            Object representing reference orientation.

        Raises
        ------
        ValueError
            Orientation with the requested index/tag not present.
        IndexError
            Orientation with specified index not found.
        """
        if isinstance(idx, str):
            num = self.get_tag_index(idx)
        else:
            num = idx - 1
        return self.orientations[num]

    def remove_orientation(self, idx: Union[str, int]) -> None:
        """Delete a reference orientation.

        Parameters
        ----------
        idx : Union[str, int]
            Index or tag of the deteled orientation.

        Raises
        ------
        ValueError
            Orientation with the requested index/tag not present.
        IndexError
            Orientation with specified index not found.
        """
        if isinstance(idx, str):
            num = self.get_tag_index(idx)
        else:
            num = idx - 1
        del self.orientations[num]

    def swap_orientations(self, idx1: Union[str, int], idx2: Union[str, int]) -> None:
        """Swap indices of two reference orientations.

        Parameters
        ----------
        idx1 : Union[str, int]
            Index or tag of the first orientation to be swapped.
        idx2 : Union[str, int]
            Index or tag of the second orientation to be swapped.

        Raises
        ------
        ValueError
            Orientation with the requested index/tag not present.
        IndexError
            Orientation with specified index not found.
        """
        if isinstance(idx1, str):
            num1 = self.get_tag_index(idx1)
        else:
            num1 = idx1 - 1
        if isinstance(idx2, str):
            num2 = self.get_tag_index(idx2)
        else:
            num2 = idx2 - 1
        orig1 = self.orientations[num1]
        self.orientations[num1] = self.orientations[num2]
        self.orientations[num2] = orig1

    def __len__(self) -> int:
        """Return number of reference orientations in the list.

        Returns
        -------
        int
            Number of reference orientationss.
        """
        return len(self.orientations)

    def __str__(self) -> str:
        """Represent the reference orientations list as a string.

        Returns
        -------
        str
            Table containing list of all orientations.
        """
        return "\n".join(self._str_lines())

    def _str_lines(self) -> List[str]:
        """Table with reference orientations data.

        Returns
        -------
        List[str]
            List containing reference orientations table rows.
        """
        axes = tuple(fd.name.upper() for fd in dataclasses.fields(Position))
        if not self.orientations:
            return ["   <<< none specified >>>"]

        lines = []

        str_format = "     %5s %5s %5s   %5s %5s %5s  " + "%8s " * len(axes) + " TAG"
        header_values = ("H", "K", "L", "X", "Y", "Z") + axes
        lines.append(str_format % header_values)

        for n in range(1, len(self.orientations) + 1):
            orient = self.get_orientation(n)
            (h, k, l), (x, y, z), angles, tag = orient.astuple
            if tag is None:
                tag = ""
            str_format = (
                "  %2d % 4.2f % 4.2f % 4.2f  "
                + "% 4.2f % 4.2f % 4.2f  "
                + "% 8.4f " * len(axes)
                + " %s"
            )
            values = (n, h, k, l, x, y, z) + angles + (tag,)
            lines.append(str_format % values)
        return lines
