from typing import Optional

import pytest
from diffcalc.hkl.geometry import Position
from diffcalc.ub.reference import (
    Orientation,
    OrientationList,
    Reflection,
    ReflectionList,
)


@pytest.fixture
def reflist() -> ReflectionList:
    return ReflectionList()


def add_example_reflection(reflist: ReflectionList, tag: Optional[str] = None) -> None:
    reflist.add_reflection(
        (0, 1, 0), Position(1, 2, 3, 4, 5, 6), 12, "" if tag is None else tag
    )


def test_str_reflection(reflist: ReflectionList):
    add_example_reflection(reflist, "ref1")

    with open(f"tests/diffcalc/ub/strings/reference/reflection.txt") as f:
        expected_string = f.read()

    assert str(reflist) == expected_string


def test_add_reflection_and_get_reflection_with_no_tag(reflist: ReflectionList):
    hkl = (0, 2, 0)
    pos = Position(1, 2, 3, 4, 5, 6)
    energy = 13

    reflist.add_reflection(hkl, pos, energy, "")
    reflection = reflist.get_reflection(1)

    assert reflection == Reflection(*hkl, pos, energy, "")


def test_add_reflection_and_get_reflection_with_tag(reflist: ReflectionList):
    hkl = (0, 1, 0)
    pos = Position(1, 2, 3, 4, 5, 6)
    energy = 12
    tag = "refl"

    reflist.add_reflection(hkl, pos, energy, tag)
    reflection = reflist.get_reflection(tag)

    assert reflection == Reflection(*hkl, pos, energy, tag)


def test_add_reflection_raises_error_for_invalid_params(reflist: ReflectionList):
    hkl = (0, 1, 0)
    pos = (1, 2, 3, 4, 5, 6)
    energy = 12

    with pytest.raises(TypeError):
        reflist.add_reflection(hkl, pos, energy, "")  # type: ignore


def test_get_reflection_raises_error_for_invalid_params(reflist: ReflectionList):
    with pytest.raises(ValueError):
        reflist.get_reflection("non-existent")

    with pytest.raises(IndexError):
        reflist.get_reflection(1)


def test_edit_reflection_and_get_reflection(reflist: ReflectionList):
    add_example_reflection(reflist)

    new_hkl = (1, 0, 1)
    new_pos = Position(7, 8, 9, 10, 11, 12)
    new_energy = 13
    new_tag = "refl"

    new_refl = Reflection(*new_hkl, new_pos, new_energy, new_tag)

    reflist.edit_reflection(1, new_hkl, new_pos, new_energy, new_tag)

    assert reflist.get_reflection(1) == new_refl
    assert reflist.get_reflection(new_tag) == new_refl


def test_swap_reflections(reflist: ReflectionList):
    hkl_two = (1, 0, 1)
    pos_two = Position(7, 8, 9, 10, 11, 12)
    energy_two = 13
    tag_two = "two"

    add_example_reflection(reflist, "one")
    reflection_one = reflist.get_reflection(1)
    reflection_two = Reflection(*hkl_two, pos_two, energy_two, tag_two)

    reflist.add_reflection(hkl_two, pos_two, energy_two, tag_two)
    reflist.swap_reflections(1, 2)

    assert reflist.get_reflection(1) == reflection_two

    reflist.swap_reflections("one", "two")

    assert reflist.get_reflection(1) == reflection_one


def test_delete_reflection(reflist: ReflectionList):
    add_example_reflection(reflist)
    reflist.remove_reflection(1)

    with pytest.raises(IndexError):
        reflist.get_reflection(1)
    with pytest.raises(IndexError):
        reflist.remove_reflection(1)


def test_serialisation_reflection(reflist: ReflectionList):
    add_example_reflection(reflist)

    serialised = reflist.asdict
    remade_reflist = ReflectionList.fromdict(serialised)

    assert remade_reflist.asdict == reflist.asdict


@pytest.fixture
def orientlist() -> OrientationList:
    return OrientationList()


def add_example_orientation(
    orientlist: OrientationList, tag: Optional[str] = None
) -> None:
    orientlist.add_orientation(
        (0, 1, 0), (0, 1, 0), Position(1, 2, 3, 4, 5, 6), "" if tag is None else tag
    )


def test_str(orientlist: OrientationList):
    add_example_orientation(orientlist, "orient")

    with open(f"tests/diffcalc/ub/strings/reference/orientation.txt") as f:
        expected_string = f.read()

    assert str(orientlist) == expected_string


def test_add_orientation_and_get_orientation_with_no_tag(orientlist: OrientationList):
    hkl = (0, 1, 0)
    xyz = (0, 1, 0)
    pos = Position(1, 2, 3, 4, 5, 6)

    orientlist.add_orientation(hkl, xyz, pos, "")
    orientation = orientlist.get_orientation(1)

    assert orientation == Orientation(*hkl, *xyz, pos, "")


def test_add_orientation_and_get_orientation_with_tag(orientlist: OrientationList):
    hkl = (0, 1, 0)
    xyz = (0, 1, 0)
    pos = Position(1, 2, 3, 4, 5, 6)
    tag = "orient"

    orientlist.add_orientation(hkl, xyz, pos, tag)
    orientation = orientlist.get_orientation(tag)

    assert orientation == Orientation(*hkl, *xyz, pos, tag)


def test_add_orientation_raises_error_for_invalid_params(orientlist: OrientationList):
    hkl = (0, 1, 0)
    xyz = (0, 1, 0)
    pos = (1, 2, 3, 4, 5, 6)

    with pytest.raises(TypeError):
        orientlist.add_orientation(hkl, xyz, pos, "")  # type: ignore


def test_get_orientation_raises_error_for_invalid_params(orientlist: OrientationList):
    with pytest.raises(ValueError):
        orientlist.get_orientation("non-existent")

    with pytest.raises(IndexError):
        orientlist.get_orientation(1)


def test_edit_orientation_and_get_orientation(orientlist: OrientationList):
    add_example_orientation(orientlist)

    new_hkl = (1, 0, 1)
    new_xyz = (1, 0, 1)
    new_pos = Position(7, 8, 9, 10, 11, 12)
    new_tag = "refl"

    new_orientation = Orientation(*new_hkl, *new_xyz, new_pos, new_tag)

    orientlist.edit_orientation(1, new_hkl, new_xyz, new_pos, new_tag)

    assert orientlist.get_orientation(1) == new_orientation
    assert orientlist.get_orientation(new_tag) == new_orientation


def test_swap_orientations(orientlist: OrientationList):
    hkl_two = (1, 0, 1)
    xyz_two = (1, 0, 1)
    pos_two = Position(7, 8, 9, 10, 11, 12)
    tag_two = "two"

    add_example_orientation(orientlist, "one")
    orientation_one = orientlist.get_orientation(1)
    orientation_two = Orientation(*hkl_two, *xyz_two, pos_two, tag_two)

    orientlist.add_orientation(hkl_two, xyz_two, pos_two, tag_two)
    orientlist.swap_orientations(1, 2)

    assert orientlist.get_orientation(1) == orientation_two

    orientlist.swap_orientations("one", "two")

    assert orientlist.get_orientation(1) == orientation_one


def test_delete_orientation(orientlist: OrientationList):
    add_example_orientation(orientlist)
    orientlist.remove_orientation(1)

    with pytest.raises(IndexError):
        orientlist.get_orientation(1)
    with pytest.raises(IndexError):
        orientlist.remove_orientation(1)


def test_serialisation_orientation(orientlist: OrientationList):
    add_example_orientation(orientlist)

    serialised = orientlist.asdict
    remade_reflist = OrientationList.fromdict(serialised)

    assert remade_reflist.asdict == orientlist.asdict
