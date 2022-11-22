from diffcalc.util import DiffcalcException

from tests.tools import angles_equivalent


class TestUtils:
    def testDegreesEqual(self):
        tol = 0.001
        assert angles_equivalent(1, 1, tol)
        assert angles_equivalent(1, -359, tol)
        assert angles_equivalent(359, -1, tol)
        assert not angles_equivalent(1.1, 1, tol)
        assert not angles_equivalent(1.1, -359, tol)
        assert not angles_equivalent(359.1, -1, tol)


def test_exception():
    exception = DiffcalcException("exception")
    assert str(exception) == "\n\n***********\n* exception\n***********"
