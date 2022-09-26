from diffcalc.hkl.geometry import Position
from diffcalc.ub.reference import OrientationList


class TestOrientationList:
    def setup_method(self):
        self.orientlist = OrientationList()
        pos = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        self.orientlist.add_orientation((1, 2, 3), (0.1, 0.2, 0.3), pos, "orient1")
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66)
        self.orientlist.add_orientation(
            (1.1, 2.2, 3.3), (0.11, 0.12, 0.13), pos, "orient2"
        )

    def test_str(self):
        res = str(self.orientlist)
        assert (
            res
            == """         H     K     L       X     Y     Z        MU    DELTA       NU      ETA      CHI      PHI  TAG
   1  1.00  2.00  3.00   0.10  0.20  0.30    0.1000   0.2000   0.3000   0.4000   0.5000   0.6000  orient1
   2  1.10  2.20  3.30   0.11  0.12  0.13    0.1100   0.2200   0.3300   0.4400   0.5500   0.6600  orient2"""
        )

    def test_add_orientation(self):
        assert len(self.orientlist) == 2
        pos = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        self.orientlist.add_orientation((10, 20, 30), (0.1, 0.2, 0.3), pos, "orient1")

    def test_get_orientation(self):
        pos = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        answered = self.orientlist.get_orientation(1).astuple
        desired = ((1, 2, 3), (0.1, 0.2, 0.3), pos.astuple, "orient1")
        assert answered == desired
        answered = self.orientlist.get_orientation("orient1").astuple
        assert answered == desired

    def testRemoveOrientation(self):
        self.orientlist.remove_orientation(1)
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66).astuple
        answered = self.orientlist.get_orientation(1).astuple
        desired = ((1.1, 2.2, 3.3), (0.11, 0.12, 0.13), pos, "orient2")
        assert answered == desired
        self.orientlist.remove_orientation("orient2")
        assert self.orientlist.orientations == []

    def testedit_orientation(self):
        ps = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        self.orientlist.edit_orientation(1, (10, 20, 30), (1, 2, 3), ps, "new1")
        assert self.orientlist.get_orientation(1).astuple == (
            (10, 20, 30),
            (1, 2, 3),
            ps.astuple,
            "new1",
        )
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66)
        assert self.orientlist.get_orientation(2).astuple == (
            (1.1, 2.2, 3.3),
            (0.11, 0.12, 0.13),
            pos.astuple,
            "orient2",
        )
        self.orientlist.edit_orientation(
            "orient2", (1.1, 2.2, 3.3), (1.11, 1.12, 1.13), pos, "new2"
        )
        assert self.orientlist.get_orientation("new2").astuple == (
            (1.1, 2.2, 3.3),
            (1.11, 1.12, 1.13),
            pos.astuple,
            "new2",
        )
        self.orientlist.edit_orientation(
            "new2", (1.1, 2.2, 3.3), (1.11, 1.12, 1.13), pos, "new1"
        )
        assert self.orientlist.get_orientation("new1").astuple == (
            (10, 20, 30),
            (1, 2, 3),
            ps.astuple,
            "new1",
        )

    def testSwapOrientation(self):
        self.orientlist.swap_orientations(1, 2)
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66)
        assert self.orientlist.get_orientation(1).astuple == (
            (1.1, 2.2, 3.3),
            (0.11, 0.12, 0.13),
            pos.astuple,
            "orient2",
        )
        pos = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        assert self.orientlist.get_orientation(2).astuple == (
            (1, 2, 3),
            (0.1, 0.2, 0.3),
            pos.astuple,
            "orient1",
        )
        self.orientlist.swap_orientations("orient1", "orient2")
        pos = Position(0.11, 0.22, 0.33, 0.44, 0.55, 0.66)
        assert self.orientlist.get_orientation(2).astuple == (
            (1.1, 2.2, 3.3),
            (0.11, 0.12, 0.13),
            pos.astuple,
            "orient2",
        )
        pos = Position(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        assert self.orientlist.get_orientation(1).astuple == (
            (1, 2, 3),
            (0.1, 0.2, 0.3),
            pos.astuple,
            "orient1",
        )

    def test_serialisation(self):
        orig_orient_list = self.orientlist
        orient_json = orig_orient_list.asdict
        reformed_orient_list = OrientationList.fromdict(orient_json)

        assert reformed_orient_list.asdict == orig_orient_list.asdict
