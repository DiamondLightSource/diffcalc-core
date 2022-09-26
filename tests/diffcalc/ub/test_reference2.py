import pytest
from diffcalc.hkl.geometry import Position
from diffcalc.ub.reference import (
    Orientation,
    OrientationList,
    Reflection,
    ReflectionList,
)

# integration tests


# unit tests


class TestReflectionList:
    reflist = ReflectionList()

    def test_add_reflection_and_get_reflection_with_no_tag(self):
        hkl = [0, 1, 0]
        pos = Position(1, 2, 3, 4, 5, 6)
        energy = 12

        self.reflist.add_reflection(hkl, pos, energy, "")
        reflection = self.reflist.get_reflection(1)

        assert reflection == Reflection(*hkl, pos, energy, "")

    def test_add_reflection_and_get_reflection_with_tag(self):
        hkl = [0, 1, 0]
        pos = Position(1, 2, 3, 4, 5, 6)
        energy = 12
        tag = "refl"

        self.reflist.add_reflection(hkl, pos, energy, tag)
        reflection = self.reflist.get_reflection(tag)

        assert reflection == Reflection(*hkl, pos, energy, tag)

    def test_add_reflection_raises_error_for_invalid_params(self):
        hkl = [0, 1, 0]
        pos = (1, 2, 3, 4, 5, 6)
        energy = 12

        with pytest.raises(TypeError):
            self.reflist.add_reflection(hkl, pos, energy, "")

    def test_get_reflection_raises_error_for_invalid_params(self):
        with pytest.raises(ValueError):
            self.reflist.get_reflection("non-existent")

        with pytest.raises(IndexError):
            self.reflist.get_reflection(1)

    def test_edit_reflection_and_get_reflection(self):
        hkl = [0, 1, 0]
        pos = Position(1, 2, 3, 4, 5, 6)
        energy = 12
        tag = ""

        new_hkl = [1, 0, 1]
        new_pos = Position(7, 8, 9, 10, 11, 12)
        new_energy = 13
        new_tag = "refl"

        self.reflist.add_reflection(hkl, pos, energy, tag)
        self.reflist.edit_reflection(0, new_hkl, new_pos, new_energy, new_tag)

        assert self.reflist.get_reflection(1) == Reflection(
            *new_hkl, new_pos, new_energy, new_tag
        )
        assert self.reflist.get_reflection(new_tag) == Reflection(
            *new_hkl, new_pos, new_energy, new_tag
        )

    def test_swap_reflections(self):
        hkl_one = [0, 1, 0]
        pos_one = Position(1, 2, 3, 4, 5, 6)
        energy_one = 12
        tag_one = "one"

        hkl_two = [1, 0, 1]
        pos_two = Position(7, 8, 9, 10, 11, 12)
        energy_two = 13
        tag_two = "two"

        self.reflist.add_reflection(hkl_one, pos_one, energy_one, tag_one)
        self.reflist.add_reflection(hkl_two, pos_two, energy_two, tag_two)

        self.reflist.swap_reflections(1, 2)

        assert self.reflist.get_reflection(1) == Reflection(
            *hkl_two, pos_two, energy_two, tag_two
        )

        self.reflist.swap_reflections("one", "two")

        assert self.reflist.get_reflection(1) == Reflection(
            *hkl_one, pos_one, energy_one, tag_one
        )

    def test_delete_reflection(self):
        hkl = [0, 1, 0]
        pos = Position(1, 2, 3, 4, 5, 6)
        energy = 12
        tag = ""

        self.reflist.add_reflection(hkl, pos, energy, tag)
        self.reflist.remove_reflection(1)

        with pytest.raises(IndexError):
            self.reflist.get_reflection(1)
        with pytest.raises(IndexError):
            self.reflist.remove_reflection(1)

    def test_serialisation(self):
        hkl = [0, 1, 0]
        pos = Position(1, 2, 3, 4, 5, 6)
        energy = 12
        tag = ""

        self.reflist.add_reflection(hkl, pos, energy, tag)

        serialised = self.reflist.asdict
        remade_reflist = ReflectionList.fromdict(serialised)

        assert remade_reflist.asdict == self.reflist.asdict


class TestOrientationList:
    orientlist = OrientationList()

    def test_add_orientation_and_get_orientation_with_no_tag(self):
        hkl = [0, 1, 0]
        xyz = [0, 1, 0]
        pos = Position(1, 2, 3, 4, 5, 6)

        self.orientlist.add_orientation(hkl, xyz, pos, "")
        orientation = self.orientlist.get_orientation(1)

        assert orientation == Orientation(*hkl, *xyz, pos, "")

    def test_add_orientation_and_get_orientation_with_tag(self):
        hkl = [0, 1, 0]
        xyz = [0, 1, 0]
        pos = Position(1, 2, 3, 4, 5, 6)
        tag = "orient"

        self.orientlist.add_orientation(hkl, xyz, pos, tag)
        orientation = self.orientlist.get_orientation(tag)

        assert orientation == Orientation(*hkl, *xyz, pos, tag)

    def test_add_orientation_raises_error_for_invalid_params(self):
        hkl = [0, 1, 0]
        xyz = [0, 1, 0]
        pos = (1, 2, 3, 4, 5, 6)

        with pytest.raises(TypeError):
            self.orientlist.add_orientation(hkl, xyz, pos, "")

    def test_get_orientation_raises_error_for_invalid_params(self):
        with pytest.raises(ValueError):
            self.orientlist.get_orientation("non-existent")

        with pytest.raises(IndexError):
            self.orientlist.get_orientation(1)

    def test_edit_orientation_and_get_orientation(self):
        hkl = [0, 1, 0]
        xyz = [0, 1, 0]
        pos = Position(1, 2, 3, 4, 5, 6)
        tag = ""

        new_hkl = [1, 0, 1]
        new_xyz = [1, 0, 1]
        new_pos = Position(7, 8, 9, 10, 11, 12)
        new_tag = "refl"

        self.orientlist.add_orientation(hkl, xyz, pos, tag)
        self.orientlist.edit_orientation(0, new_hkl, new_xyz, new_pos, new_tag)

        assert self.orientlist.get_orientation(1) == Orientation(
            *new_hkl, *new_xyz, new_pos, new_tag
        )
        assert self.orientlist.get_orientation(new_tag) == Orientation(
            *new_hkl, *new_xyz, new_pos, new_tag
        )

    def test_swap_reflections(self):
        hkl_one = [0, 1, 0]
        xyz_one = [0, 1, 0]
        pos_one = Position(1, 2, 3, 4, 5, 6)
        tag_one = "one"

        hkl_two = [1, 0, 1]
        xyz_two = [1, 0, 1]
        pos_two = Position(7, 8, 9, 10, 11, 12)
        tag_two = "two"

        self.orientlist.add_orientation(hkl_one, xyz_one, pos_one, tag_one)
        self.orientlist.add_orientation(hkl_two, xyz_two, pos_two, tag_two)

        self.orientlist.swap_orientations(1, 2)

        assert self.orientlist.get_orientation(1) == Orientation(
            *hkl_two, *xyz_two, pos_two, tag_two
        )

        self.orientlist.swap_orientations("one", "two")

        assert self.orientlist.get_orientation(1) == Orientation(
            *hkl_one, *xyz_one, pos_one, tag_one
        )

    def test_delete_reflection(self):
        hkl = [0, 1, 0]
        xyz = [0, 1, 0]
        pos = Position(1, 2, 3, 4, 5, 6)
        tag = ""

        self.orientlist.add_orientation(hkl, xyz, pos, tag)
        self.orientlist.remove_orientation(1)

        with pytest.raises(IndexError):
            self.orientlist.get_orientation(1)
        with pytest.raises(IndexError):
            self.orientlist.remove_orientation(1)

    def test_serialisation(self):
        hkl = [0, 1, 0]
        xyz = [0, 1, 0]
        pos = Position(1, 2, 3, 4, 5, 6)
        tag = ""

        self.orientlist.add_orientation(hkl, xyz, pos, tag)

        serialised = self.orientlist.asdict
        remade_reflist = OrientationList.fromdict(serialised)

        assert remade_reflist.asdict == self.orientlist.asdict
