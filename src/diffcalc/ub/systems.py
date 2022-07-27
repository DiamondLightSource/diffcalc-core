from enum import Enum
from math import pi
from typing import Dict, Optional, Union

SystemType = Dict[str, Optional[Union[float, str]]]


class Systems(Enum):
    Triclinic: SystemType = {
        "a": None,
        "b": None,
        "c": None,
        "alpha": None,
        "beta": None,
        "gamma": None,
    }
    Monoclinic: SystemType = {
        "a": None,
        "b": None,
        "c": None,
        "alpha": pi / 2,
        "beta": None,
        "gamma": pi / 2,
    }
    Orthorhombic: SystemType = {
        "a": None,
        "b": None,
        "c": None,
        "alpha": pi / 2,
        "beta": pi / 2,
        "gamma": pi / 2,
    }
    Tetragonal: SystemType = {
        "a": None,
        "b": "a",
        "c": None,
        "alpha": pi / 2,
        "beta": pi / 2,
        "gamma": pi / 2,
    }
    Rhombohedral: SystemType = {
        "a": None,
        "b": "a",
        "c": "a",
        "alpha": None,
        "beta": "alpha",
        "gamma": "alpha",
    }
    Hexagonal: SystemType = {
        "a": None,
        "b": "a",
        "c": None,
        "alpha": pi / 2,
        "beta": pi / 2,
        "gamma": 2 * pi / 3,
    }
    Cubic: SystemType = {
        "a": None,
        "b": "a",
        "c": "a",
        "alpha": pi / 2,
        "beta": pi / 2,
        "gamma": pi / 2,
    }


available_systems = [item for item in dir(Systems) if not item.startswith("_")]
