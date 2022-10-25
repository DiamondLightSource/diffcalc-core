from diffcalc.util import DiffcalcException

from tests.tools import degrees_equivalent


class TestUtils:
    def testDegreesEqual(self):
        tol = 0.001
        assert degrees_equivalent(1, 1, tol)
        assert degrees_equivalent(1, -359, tol)
        assert degrees_equivalent(359, -1, tol)
        assert not degrees_equivalent(1.1, 1, tol)
        assert not degrees_equivalent(1.1, -359, tol)
        assert not degrees_equivalent(359.1, -1, tol)


def test_exception():
    exception = DiffcalcException("exception")
    assert str(exception) == "\n\n***********\n* exception\n***********"
