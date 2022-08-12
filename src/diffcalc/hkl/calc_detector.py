"""Module implementing intermediate calculations in constrained detector geometry."""
from itertools import product
from math import acos, asin, atan2, cos, degrees, isnan, pi, sin
from typing import Dict, Iterator, Optional, Tuple

from diffcalc.util import DiffcalcException, bound, is_small, sign


def _calc_angle_between_naz_and_qaz(theta: float, alpha: float, tau: float) -> float:
    # Equation 30:
    top = cos(tau) - sin(alpha) * sin(theta)
    bottom = cos(alpha) * cos(theta)
    if is_small(bottom):
        if is_small(cos(alpha)):
            return float("nan")
    if isnan(tau) or is_small(sin(tau)):
        return 0.0
    return acos(bound(top / bottom))


def _calc_remaining_detector_angles_delta(
    delta: float, theta: float
) -> Iterator[Tuple[float, float, float]]:
    """Return delta, nu and qaz given delta detector angle."""
    #                                                         (section 5.1)
    # Find qaz using various derivations of 17 and 18
    try:
        asin_qaz = asin(bound(sin(delta) / sin(2.0 * theta)))  # (17 & 18)
    except AssertionError:
        return
    cos_delta = cos(delta)
    if is_small(cos_delta):
        # raise DiffcalcException(
        #    'The %s and %s circles are redundant when delta is constrained to %.0f degrees.'
        #    'Please change delta constraint or use 4-circle mode.' % ("nu", 'mu', delta * TODEG))
        print(
            (
                "DEGENERATE: with delta=90, %s is degenerate: choosing "
                "%s = 0 (allowed because %s is unconstrained)."
            )
            % ("nu", "nu", "nu")
        )
        acos_nu = 0.0
    else:
        try:
            acos_nu = acos(bound(cos(2.0 * theta) / cos_delta))
        except AssertionError:
            return
    if is_small(cos(asin_qaz)):
        qaz_angles = [
            sign(asin_qaz) * pi / 2.0,
        ]
    else:
        qaz_angles = [asin_qaz, pi - asin_qaz]
    if is_small(acos_nu):
        nu_angles = [
            0.0,
        ]
    else:
        nu_angles = [acos_nu, -acos_nu]
    for qaz, nu in product(qaz_angles, nu_angles):
        sgn_ref = sign(sin(2.0 * theta)) * sign(cos(qaz))
        sgn_ratio = sign(sin(nu)) * sign(cos_delta)
        if sgn_ref == sgn_ratio:
            yield delta, nu, qaz


def _calc_remaining_detector_angles_nu(
    nu: float, theta: float
) -> Iterator[Tuple[float, float, float]]:
    """Return delta, nu and qaz given nu detector angle."""
    #                                                         (section 5.1)
    # Find qaz using various derivations of 17 and 18
    sin_2theta = sin(2 * theta)
    cos_2theta = cos(2 * theta)
    cos_nu = cos(nu)
    if is_small(cos_nu):
        raise DiffcalcException(
            "The %s circle constraint to %.0f degrees is redundant."
            "Please change this constraint or use 4-circle mode." % ("nu", degrees(nu))
        )
    cos_delta = cos_2theta / cos(nu)
    cos_qaz = cos_delta * sin(nu) / sin_2theta
    try:
        acos_delta = acos(bound(cos_delta))
        acos_qaz = acos(bound(cos_qaz))
    except AssertionError:
        return
    if is_small(acos_qaz):
        qaz_angles = [
            0.0,
        ]
    else:
        qaz_angles = [acos_qaz, -acos_qaz]
    if is_small(acos_delta):
        delta_angles = [
            0.0,
        ]
    else:
        delta_angles = [acos_delta, -acos_delta]
    for qaz, delta in product(qaz_angles, delta_angles):
        sgn_ref = sign(sin(delta))
        sgn_ratio = sign(sin(qaz)) * sign(sin_2theta)
        if sgn_ref == sgn_ratio:
            yield delta, nu, qaz


def _calc_remaining_detector_angles_qaz(
    qaz: float, theta: float
) -> Iterator[Tuple[float, float, float]]:
    """Return delta, nu and qaz given qaz detector angle."""
    #                                                         (section 5.1)
    # Find qaz using various derivations of 17 and 18
    sin_2theta = sin(2 * theta)
    cos_2theta = cos(2 * theta)
    asin_delta = asin(sin(qaz) * sin_2theta)
    if is_small(cos(asin_delta)):
        delta_angles = [
            sign(asin_delta) * pi / 2.0,
        ]
    else:
        delta_angles = [asin_delta, pi - asin_delta]
    for delta in delta_angles:
        cos_delta = cos(delta)
        if is_small(cos_delta):
            print(
                (
                    "DEGENERATE: with delta=90, %s is degenerate: choosing "
                    "%s = 0 (allowed because %s is unconstrained)."
                )
                % ("nu", "nu", "nu")
            )
            # raise DiffcalcException(
            #    'The %s circle is redundant when delta is at %.0f degrees.'
            #    'Please change detector constraint or use 4-circle mode.' % ("nu", delta * TODEG))
            nu = 0.0
        else:
            sgn_delta = sign(cos_delta)
            nu = atan2(sgn_delta * sin_2theta * cos(qaz), sgn_delta * cos_2theta)
        yield delta, nu, qaz


def _calc_detector_con_det_or_naz(
    det_constraint: Dict[str, Optional[float]],
    naz_constraint: Dict[str, Optional[float]],
    theta: float,
    tau: float,
    alpha: float,
) -> Iterator[Tuple[float, float, float, float]]:

    assert det_constraint or naz_constraint
    try:
        naz_qaz_angle = _calc_angle_between_naz_and_qaz(theta, alpha, tau)
    except AssertionError:
        return
    _calc_remaining_detector_angles = {
        "delta": _calc_remaining_detector_angles_delta,
        "nu": _calc_remaining_detector_angles_nu,
        "qaz": _calc_remaining_detector_angles_qaz,
    }
    if det_constraint:
        # One of the detector angles is given                 (Section 5.1)
        det_constraint_name, det_constraint_value = next(iter(det_constraint.items()))
        for delta, nu, qaz in _calc_remaining_detector_angles[det_constraint_name](
            det_constraint_value, theta
        ):
            if is_small(naz_qaz_angle):
                naz_angles = [
                    qaz,
                ]
            elif isnan(naz_qaz_angle):
                naz_angles = [float("nan")]
            else:
                naz_angles = [qaz - naz_qaz_angle, qaz + naz_qaz_angle]
            for naz in naz_angles:
                yield qaz, naz, delta, nu
    elif naz_constraint:  # The 'detector' angle naz is given:
        naz_name, naz = next(iter(naz_constraint.items()))
        assert naz_name == "naz"
        if is_small(naz_qaz_angle):
            qaz_angles = [
                naz,
            ]
        else:
            qaz_angles = [naz - naz_qaz_angle, naz + naz_qaz_angle]
        for qaz in qaz_angles:
            for delta, nu, _ in _calc_remaining_detector_angles_qaz(qaz, theta):
                yield qaz, naz, delta, nu
