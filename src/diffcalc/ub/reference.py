"""Module providing objects for working with reference reflections and orientations."""
import dataclasses
from typing import Any, Dict, List, Tuple, Union

from diffcalc.hkl.geometry import Position


@dataclasses.dataclass
class Reference:
    h: float
    k: float
    l: float
    pos: Position
    tag: str

    @property
    def asdict(self) -> Dict[str, Any]:
        return {}

    @property
    def astuple(self) -> Tuple[Any, ...]:
        return ()


@dataclasses.dataclass
class Reflection(Reference):
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

    energy: float

    @property
    def astuple(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[Any, ...], float, str,]:
        """Return reference reflection data as tuple.

        Returns
        -------
        Tuple[Tuple[float, float, float],
              Tuple[Any, ...],
              float,
              str]
            Tuple containing miller indices, position object, energy and
            reflection tag.
        """
        h, k, l, pos, tag, en = dataclasses.astuple(self)
        return (h, k, l), pos, tag, en

    @property
    def asdict(self) -> Dict[str, Any]:
        """Return reference reflection data as dictionary.

        Returns
        -------
        JSONReflection
            Class structure containing miller indices, position as a dictionary, energy
            and reflection tag.
        """
        class_info = self.__dict__.copy()
        class_info["pos"] = self.pos.asdict
        return class_info

    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> Reference:
        """Create reflection object from a dictionary.

        Parameters
        ----------
        data : JSONReflection
            Class structure containing miller indices, position as a dictionary, energy
            and reflection tag

        Returns
        -------
        Reflection
            An instance of the Reflection class.
        """
        return cls(
            data["h"],
            data["k"],
            data["l"],
            Position(**data["pos"]),
            data["tag"],
            data["energy"],
        )


@dataclasses.dataclass
class Orientation(Reference):
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

    x: float
    y: float
    z: float

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
        h, k, l, pos, tag, x, y, z = dataclasses.astuple(self)
        return (h, k, l), (x, y, z), pos, tag

    @property
    def asdict(self) -> Dict[str, Any]:
        class_info = self.__dict__.copy()
        class_info["pos"] = self.pos.asdict
        return class_info

    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> Reference:
        return cls(
            data["h"],
            data["k"],
            data["l"],
            Position(**data["pos"]),
            data["tag"],
            data["x"],
            data["y"],
            data["z"],
        )


@dataclasses.dataclass
class RefOrientList:
    items: List[Reference] = dataclasses.field(default_factory=list)

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
        _tag_list = [ref.tag for ref in self.items]
        num = _tag_list.index(tag)
        return num

    def get_item(self, idx: Union[str, int]) -> Reference:
        """Get item from list of reference reflections/orientations.

        Get aon object representing reference reflection.

        Parameters
        ----------
        idx : Union[str, int]
            Index or tag of the reflection. Index same as python arrays; starts at 0

        Returns
        -------
        Reflection
            Object representing reference reflection

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
            num = idx
        return self.items[num]

    def remove_item(self, idx: Union[str, int]) -> None:
        """Remove item from list of reference reflections/orientations.

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
        del self.items[num]

    def swap_items(self, idx1: Union[str, int], idx2: Union[str, int]) -> None:
        """Swap indices of two items from list of reference reflections/orientations.

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
        orig1 = self.items[num1]
        self.items[num1] = self.items[num2]
        self.items[num2] = orig1

    @property
    def asdict(self) -> List[Dict[str, Any]]:
        return [ref.asdict for ref in self.items]

    def __len__(self) -> int:
        """Return number of reference reflections in the list.

        Returns
        -------
        int
            Number of reference reflections.
        """
        return len(self.items)

    def __str__(self) -> str:
        """Represent the reference reflection list as a string.

        Returns
        -------
        str
            Table containing list of all reflections.
        """
        return "\n".join(self._str_lines())

    def _str_lines(self) -> List[str]:
        return []


class ReflectionList(RefOrientList):
    """Class containing collection of reference reflections.

    Attributes
    ----------
    reflections: List[Reflection]
        List containing reference reflections.
    """

    def add_item(
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
        self.items += [Reflection(*hkl, pos, tag, energy)]

    def edit_item(
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

        self.items[num] = Reflection(*hkl, pos, tag, energy)

    @classmethod
    def fromdict(cls, data: List[Dict[str, Any]]) -> "ReflectionList":
        reflections = [Reflection.fromdict(each_ref) for each_ref in data]
        return cls(reflections)

    def _str_lines(self) -> List[str]:
        """Table with reference reflection data.

        Returns
        -------
        List[str]
            List containing reference reflection table rows.
        """
        axes = tuple(fd.name.upper() for fd in dataclasses.fields(Position))
        if not self.items:
            return ["   <<< none specified >>>"]

        lines = []

        fmt = "     %6s %5s %5s %5s  " + "%8s " * (len(axes) - 1) + " %4s" + " %4s"
        header_values = ("ENERGY", "H", "K", "L") + axes + ("TAG",)
        lines.append(fmt % header_values)

        for n in range(len(self.items)):
            ref_tuple = self.get_item(n)
            (h, k, l), pos, tag, energy = ref_tuple.astuple
            if tag is None:
                tag = ""
            fmt = (
                "  %2d %6.3f % 4.2f % 4.2f % 4.2f  "
                + "% 8.4f " * (len(axes) - 1)
                + " %4r"
                + " %9s"
            )
            values = (n, energy, h, k, l) + pos + (tag,)
            lines.append(fmt % values)
        return lines


class OrientationList(RefOrientList):
    """Class containing collection of reference orientations.

    Attributes
    ----------
    reflections: List[Orientation]
        List containing reference orientations.
    """

    def add_item(
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
            self.items += [Orientation(*hkl, pos, tag, *xyz)]
        else:
            raise TypeError("Invalid position parameter type")

    def edit_item(
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

        self.items[num] = Orientation(*hkl, pos, tag, *xyz)

    @classmethod
    def fromdict(cls, data: List[Dict[str, Any]]) -> "OrientationList":
        orientations = [Orientation.fromdict(each_ref) for each_ref in data]
        return cls(orientations)

    def _str_lines(self) -> List[str]:
        """Table with reference orientations data.

        Returns
        -------
        List[str]
            List containing reference orientations table rows.
        """
        axes = tuple(fd.name.upper() for fd in dataclasses.fields(Position))
        if not self.items:
            return ["   <<< none specified >>>"]

        lines = []

        str_format = (
            "     %5s %5s %5s   %5s %5s %5s"
            + " %9s " * (len(axes) - 1)
            + " %4s"
            + " %4s"
        )
        header_values = ("H", "K", "L", "X", "Y", "Z") + axes + ("TAG",)
        lines.append(str_format % header_values)

        for n in range(len(self.items)):
            orient = self.get_item(n)
            (h, k, l), (x, y, z), angles, tag = orient.astuple
            if tag is None:
                tag = ""
            str_format = (
                "  %5d % 5.2f % 5.2f % 5.2f  "
                + " %5.2f % 5.2f % 5.2f "
                + " %9.4f" * (len(axes) - 1)
                + " %8r"
                + " %s"
            )
            values = (n, h, k, l, x, y, z) + angles + (tag,)
            lines.append(str_format % values)
        return lines
