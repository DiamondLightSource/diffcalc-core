"""Module implementing intermediate calculations in constrained sample geometry."""
import logging
from itertools import product
from math import acos, asin, atan, atan2, cos, degrees, isnan, pi, sin, sqrt, tan
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from diffcalc.hkl.geometry import rot_CHI, rot_ETA, rot_MU, rot_PHI
from diffcalc.util import (
    DiffcalcException,
    angle_between_vectors,
    bound,
    cross3,
    is_small,
    normalised,
    sign,
    y_rotation,
    z_rotation,
)
from numpy.linalg import inv, norm


def _calc_N(Q: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Return N as described by Equation 31."""
    Q = normalised(Q)
    n = normalised(n)
    if is_small(angle_between_vectors(Q, n)):
        # Replace the reference vector with an alternative vector from Eq.(78)
        def __key_func(v):
            return v[1]  # Workaround for mypy issue #9590

        idx_min, _ = min(
            enumerate([abs(Q[0, 0]), abs(Q[1, 0]), abs(Q[2, 0])]),
            key=__key_func,
        )
        idx_1, idx_2 = (idx for idx in range(3) if idx != idx_min)
        qval = sqrt(Q[idx_1, 0] * Q[idx_1, 0] + Q[idx_2, 0] * Q[idx_2, 0])
        n[idx_min, 0] = qval
        n[idx_1, 0] = -Q[idx_min, 0] * Q[idx_1, 0] / qval
        n[idx_2, 0] = -Q[idx_min, 0] * Q[idx_2, 0] / qval
        if is_small(norm(n)):
            n[idx_min, 0] = 0
            n[idx_1, 0] = Q[idx_2, 0] / qval
            n[idx_2, 0] = -Q[idx_1, 0] / qval
    Qxn = cross3(Q, n)
    QxnxQ = cross3(Qxn, Q)
    QxnxQ = normalised(QxnxQ)
    Qxn = normalised(Qxn)
    return np.array(
        [
            [Q[0, 0], QxnxQ[0, 0], Qxn[0, 0]],
            [Q[1, 0], QxnxQ[1, 0], Qxn[1, 0]],
            [Q[2, 0], QxnxQ[2, 0], Qxn[2, 0]],
        ]
    )


def __calc_sample_con_mu(
    samp_constraints: Dict[str, Optional[float]],
    N_lab: np.ndarray,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    mu = samp_constraints["mu"]
    V = inv(rot_MU(mu)) @ N_lab @ N_phi.T
    try:
        acos_chi = acos(bound(V[2, 2]))
    except AssertionError:
        return
    if is_small(sin(acos_chi)):
        # chi ~= 0 or 180 and therefor phi || eta The solutions for phi
        # and eta here will be valid but will be chosen unpredictably.
        # Choose eta=0:
        #
        # tan(phi+eta)=v12/v11 from docs/extensions_to_yous_paper.wxm
        chi = acos_chi
        eta = 0.0
        phi = atan2(-V[1, 0], V[1, 1])
        logging.debug(
            "Eta and phi cannot be chosen uniquely with chi so close "
            "to 0 or 180. Returning phi=%.3f and eta=%.3f",
            degrees(phi),
            degrees(eta),
        )
        yield mu, eta, chi, phi
    else:
        for chi in [acos_chi, -acos_chi]:
            sgn = sign(sin(chi))
            phi = atan2(-sgn * V[2, 1], -sgn * V[2, 0])
            eta = atan2(-sgn * V[1, 2], sgn * V[0, 2])
            yield mu, eta, chi, phi


def __calc_sample_con_phi(
    samp_constraints: Dict[str, Optional[float]],
    N_lab: np.ndarray,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    phi = samp_constraints["phi"]
    V = N_lab @ inv(N_phi) @ rot_PHI(phi).T
    try:
        asin_eta = asin(bound(V[0, 1]))
    except AssertionError:
        return
    if is_small(cos(asin_eta)):
        raise DiffcalcException(
            "Chi and mu cannot be chosen uniquely " "with eta so close to +/-90."
        )
    for eta in [asin_eta, pi - asin_eta]:
        sgn = sign(cos(eta))
        mu = atan2(sgn * V[2, 1], sgn * V[1, 1])
        chi = atan2(sgn * V[0, 2], sgn * V[0, 0])
        yield mu, eta, chi, phi


def __calc_sample_from_chi_eta(
    chi: float,
    eta: float,
    Z: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    top_for_mu = Z[2, 2] * sin(eta) * sin(chi) + Z[1, 2] * cos(chi)
    bot_for_mu = -Z[2, 2] * cos(chi) + Z[1, 2] * sin(eta) * sin(chi)
    if is_small(top_for_mu) and is_small(bot_for_mu):
        # chi == +-90, eta == 0/180 and therefore phi || mu cos(chi) ==
        # 0 and sin(eta) == 0 Experience shows that even though e.g.
        # the z[2, 2] and z[1, 2] values used to calculate mu may be
        # basically 0 (1e-34) their ratio in every case tested so far
        # still remains valid and using them will result in a phi
        # solution that is continuous with neighbouring positions.
        #
        # We cannot test phi minus mu here unfortunately as the final
        # phi and mu solutions have not yet been chosen (they may be
        # +-x or 180+-x). Otherwise we could choose a sensible solution
        # here if the one found was incorrect.

        # tan(phi+eta)=v12/v11 from extensions_to_yous_paper.wxm
        # phi_minus_mu = -atan2(Z[2, 0], Z[1, 1])
        raise DiffcalcException(
            "Mu cannot be chosen uniquely as mu || phi with chi so close "
            "to +/-90 and eta so close 0 or 180.\nPlease choose "
            "a different set of constraints."
        )
    mu = atan2(-top_for_mu, -bot_for_mu)  # (41)

    top_for_phi = Z[0, 1] * cos(eta) * cos(chi) - Z[0, 0] * sin(eta)
    bot_for_phi = Z[0, 1] * sin(eta) + Z[0, 0] * cos(eta) * cos(chi)
    if is_small(bot_for_phi) and is_small(top_for_phi):
        DiffcalcException(
            "Phi cannot be chosen uniquely as mu || phi with chi so close "
            "to +/-90 and eta so close 0 or 180.\nPlease choose a "
            "different set of constraints."
        )
    phi = atan2(top_for_phi, bot_for_phi)  # (42)
    yield mu, eta, chi, phi


def __calc_sample_con_chi(
    samp_constraints: Dict[str, Optional[float]],
    N_lab: np.ndarray,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    chi = samp_constraints["chi"]
    sin_chi = sin(chi)
    if is_small(sin_chi):
        raise DiffcalcException(
            "Eta and phi cannot be chosen uniquely with chi "
            "constrained so close to 0. (Please contact developer "
            "if this case is useful for you)."
        )
    Z = N_lab @ N_phi.T
    try:
        acos_eta = acos(bound(Z[0, 2] / sin_chi))
    except AssertionError:
        return
    for eta in [acos_eta, -acos_eta]:
        yield from __calc_sample_from_chi_eta(chi, eta, Z)


def __calc_sample_con_eta(
    samp_constraints: Dict[str, Optional[float]],
    N_lab: np.ndarray,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    eta = samp_constraints["eta"]
    cos_eta = cos(eta)
    if is_small(cos_eta):
        # TODO: Not likely to happen in real world!?
        raise DiffcalcException(
            "Chi and mu cannot be chosen uniquely with eta "
            "constrained so close to +-90."
        )
    Z = N_lab @ N_phi.T
    try:
        asin_chi = asin(bound(Z[0, 2] / cos_eta))
    except AssertionError:
        return
    for chi in [asin_chi, pi - asin_chi]:
        yield from __calc_sample_from_chi_eta(chi, eta, Z)


def _calc_remaining_sample_angles(
    samp_constraints: Dict[str, Optional[float]],
    theta: float,
    alpha: float,
    qaz: float,
    naz: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    """Return phi, chi, eta and mu, given one of these."""
    #                                                         (section 5.3)

    constraint_name = next(iter(samp_constraints.keys()))
    q_lab = np.array(
        [[cos(theta) * sin(qaz)], [-sin(theta)], [cos(theta) * cos(qaz)]]
    )  # (18)
    if isnan(naz):
        n_lab = np.array([[0.0], [-sin(alpha)], [0.0]])
    else:
        n_lab = np.array(
            [[cos(alpha) * sin(naz)], [-sin(alpha)], [cos(alpha) * cos(naz)]]
        )  # (20)

    N_lab = _calc_N(q_lab, n_lab)

    if constraint_name == "mu":  # (35)
        yield from __calc_sample_con_mu(samp_constraints, N_lab, N_phi)
    elif constraint_name == "phi":  # (37)
        yield from __calc_sample_con_phi(samp_constraints, N_lab, N_phi)
    elif constraint_name == "eta":  # (39)
        yield from __calc_sample_con_eta(samp_constraints, N_lab, N_phi)
    elif constraint_name == "chi":  # (40)
        yield from __calc_sample_con_chi(samp_constraints, N_lab, N_phi)
    else:
        raise DiffcalcException("Given angle must be one of phi, chi, eta or mu")


def __calc_sample_con_mu_eta(
    mu: float,
    eta: float,
    qaz: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    F = y_rotation(qaz - pi / 2.0)
    THETA = z_rotation(-theta)
    V = rot_ETA(eta).T @ rot_MU(mu).T @ F @ THETA  # (56)

    phi_vals = []
    try:
        # For the case of (00l) reflection, where N_phi[0,0] = N_phi[1,0] = 0
        if is_small(N_phi[0, 0]) and is_small(N_phi[1, 0]):
            raise DiffcalcException(
                "Phi cannot be chosen uniquely as q || phi and no reference "
                "vector or phi constraints have been set.\nPlease choose a different "
                "set of constraints."
            )
        bot = bound(-V[1, 0] / sqrt(N_phi[0, 0] ** 2 + N_phi[1, 0] ** 2))
        eps = atan2(N_phi[1, 0], N_phi[0, 0])
        phi_vals = [asin(bot) + eps, pi - asin(bot) + eps]  # (59)
    except AssertionError:
        return
    for phi in phi_vals:
        a = N_phi[0, 0] * cos(phi) + N_phi[1, 0] * sin(phi)
        chi = atan2(
            N_phi[2, 0] * V[0, 0] - a * V[2, 0],
            N_phi[2, 0] * V[2, 0] + a * V[0, 0],
        )  # (60)
        yield mu, eta, chi, phi


def __calc_sample_con_omega_bisect(
    samp_constraints: Dict[str, Optional[float]],
    qaz: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    omega = samp_constraints["omega"]
    atan_mu = atan(tan(theta + omega) * cos(qaz))
    asin_eta = asin(sin(theta + omega) * sin(qaz))
    mu_vals = [atan_mu, atan_mu + pi]
    if is_small(abs(asin_eta) - pi / 2):
        eta_vals = [
            sign(asin_eta) * pi / 2,
        ]
    else:
        eta_vals = [asin_eta, pi - asin_eta]
    for mu, eta in product(mu_vals, eta_vals):
        yield from __calc_sample_con_mu_eta(mu, eta, qaz, theta, N_phi)


def __calc_sample_con_mu_bisect(
    samp_constraints: Dict[str, Optional[float]],
    qaz: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    mu_vals = [
        samp_constraints["mu"],
    ]
    cos_qaz = cos(qaz)
    tan_mu = tan(samp_constraints["mu"])
    # Vertical scattering geometry with omega = 0
    if is_small(cos_qaz):
        if is_small(tan_mu):
            thomega_vals = [
                theta,
            ]
        else:
            return
    else:
        atan_thomega = atan(tan_mu / cos_qaz)
        thomega_vals = [atan_thomega, pi + atan_thomega]
    eta_vals = []
    for thomega in thomega_vals:
        asin_eta = asin(sin(thomega) * sin(qaz))
        if is_small(abs(asin_eta) - pi / 2):
            eta_vals.extend(
                [
                    sign(asin_eta) * pi / 2,
                ]
            )
        else:
            eta_vals.extend([asin_eta, pi - asin_eta])
    for mu, eta in product(mu_vals, eta_vals):
        yield from __calc_sample_con_mu_eta(mu, eta, qaz, theta, N_phi)


def __calc_sample_con_eta_bisect(
    samp_constraints: Dict[str, Optional[float]],
    qaz: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    eta_vals = [
        samp_constraints["eta"],
    ]
    sin_qaz = sin(qaz)
    sin_eta = sin(samp_constraints["eta"])
    # Horizontal scattering geometry with omega = 0
    if is_small(sin_qaz):
        if is_small(sin_eta):
            thomega_vals = [
                theta,
            ]
        else:
            return
    else:
        asin_thomega = asin(sin_eta / sin_qaz)
        if is_small(abs(asin_thomega) - pi / 2):
            thomega_vals = [
                sign(asin_thomega) * pi / 2,
            ]
        else:
            thomega_vals = [asin_thomega, pi - asin_thomega]
    mu_vals = []
    for thomega in thomega_vals:
        atan_mu = atan(tan(thomega) * cos(qaz))
        mu_vals.extend([atan_mu, pi + atan_mu])
    for mu, eta in product(mu_vals, eta_vals):
        yield from __calc_sample_con_mu_eta(mu, eta, qaz, theta, N_phi)


def __calc_sample_con_chi_phi(
    samp_constraints: Dict[str, Optional[float]],
    qaz: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    chi = samp_constraints["chi"]
    phi = samp_constraints["phi"]

    CHI = rot_CHI(chi)
    PHI = rot_PHI(phi)
    V = CHI @ PHI @ N_phi  # (62)

    try:
        bot = bound(V[2, 0] / sqrt(cos(qaz) ** 2 * cos(theta) ** 2 + sin(theta) ** 2))
    except AssertionError:
        return
    eps = atan2(-cos(qaz) * cos(theta), sin(theta))
    for mu in [asin(bot) + eps, pi - asin(bot) + eps]:
        a = cos(theta) * sin(qaz)
        b = -cos(theta) * sin(mu) * cos(qaz) + cos(mu) * sin(theta)
        X = V[1, 0] * a + V[0, 0] * b
        Y = V[0, 0] * a - V[1, 0] * b
        if is_small(X) and is_small(Y):
            raise DiffcalcException(
                "Eta cannot be chosen uniquely as q || eta and no reference "
                "vector or eta constraints have been set.\nPlease choose a different "
                "set of constraints."
            )
        eta = atan2(X, Y)

        # a = -cos(mu) * cos(qaz) * sin(theta) + sin(mu) * cos(theta)
        # b = cos(mu) * sin(qaz)
        # psi = atan2(-V[2, 2] * a - V[2, 1] * b, V[2, 1] * a - V[2, 2] * b)
        yield mu, eta, chi, phi


def __calc_sample_con_mu_phi(
    samp_constraints: Dict[str, Optional[float]],
    qaz: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    mu = samp_constraints["mu"]
    phi = samp_constraints["phi"]

    F = y_rotation(qaz - pi / 2.0)
    THETA = z_rotation(-theta)
    V = rot_MU(mu).T @ F @ THETA
    E = rot_PHI(phi) @ N_phi

    try:
        bot = bound(-V[2, 0] / sqrt(E[0, 0] ** 2 + E[2, 0] ** 2))
    except AssertionError:
        return
    eps = atan2(E[2, 0], E[0, 0])
    for chi in [asin(bot) + eps, pi - asin(bot) + eps]:
        a = E[0, 0] * cos(chi) + E[2, 0] * sin(chi)
        eta = atan2(V[0, 0] * E[1, 0] - V[1, 0] * a, V[0, 0] * a + V[1, 0] * E[1, 0])
        yield mu, eta, chi, phi


def __calc_sample_con_mu_chi(
    samp_constraints: Dict[str, Optional[float]],
    qaz: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    mu = samp_constraints["mu"]
    chi = samp_constraints["chi"]

    V20 = cos(mu) * cos(qaz) * cos(theta) + sin(mu) * sin(theta)
    A = N_phi[1, 0]
    B = N_phi[0, 0]
    if is_small(sin(chi)):
        raise DiffcalcException(
            "Degenerate configuration with phi || eta axes cannot be set uniquely.\n"
            "Please choose a different set of constraints."
        )
    if is_small(A) and is_small(B):
        raise DiffcalcException(
            "Phi cannot be chosen uniquely. Please choose a different set of constraints."
        )
    else:
        ks = atan2(A, B)
    try:
        acos_phi = acos(
            bound((N_phi[2, 0] * cos(chi) - V20) / (sin(chi) * sqrt(A**2 + B**2)))
        )
    except AssertionError:
        return
    if is_small(acos_phi):
        phi_list = [
            ks,
        ]
    else:
        phi_list = [acos_phi + ks, -acos_phi + ks]
    for phi in phi_list:
        A00 = -cos(qaz) * cos(theta) * sin(mu) + cos(mu) * sin(theta)
        B00 = sin(qaz) * cos(theta)
        V00 = (
            N_phi[0, 0] * cos(chi) * cos(phi)
            + N_phi[1, 0] * cos(chi) * sin(phi)
            + N_phi[2, 0] * sin(chi)
        )
        V10 = N_phi[1, 0] * cos(phi) - N_phi[0, 0] * sin(phi)
        sin_eta = V00 * A00 + V10 * B00
        cos_eta = V00 * B00 - V10 * A00
        if is_small(A00) and is_small(B00):
            raise DiffcalcException(
                "Eta cannot be chosen uniquely. Please choose a different set of constraints."
            )
        eta = atan2(sin_eta, cos_eta)
        yield mu, eta, chi, phi


def __calc_sample_con_eta_phi(
    samp_constraints: Dict[str, Optional[float]],
    qaz: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    eta = samp_constraints["eta"]
    phi = samp_constraints["phi"]

    X = N_phi[2, 0]
    Y = N_phi[0, 0] * cos(phi) + N_phi[1, 0] * sin(phi)
    if is_small(X) and is_small(Y):
        raise DiffcalcException(
            "Chi cannot be chosen uniquely as q || chi and no reference "
            "vector or chi constraints have been set.\nPlease choose a different "
            "set of constraints."
        )

    V = (N_phi[1, 0] * cos(phi) - N_phi[0, 0] * sin(phi)) * tan(eta)
    sgn = sign(cos(eta))
    eps = atan2(X * sgn, Y * sgn)
    try:
        acos_rhs = acos(
            bound((sin(qaz) * cos(theta) / cos(eta) - V) / sqrt(X**2 + Y**2))
        )
    except AssertionError:
        return
    if is_small(acos_rhs):
        acos_list = [
            eps,
        ]
    else:
        acos_list = [eps + acos_rhs, eps - acos_rhs]
    for chi in acos_list:
        A = (N_phi[0, 0] * cos(phi) + N_phi[1, 0] * sin(phi)) * sin(chi) - N_phi[
            2, 0
        ] * cos(chi)
        B = (
            -N_phi[2, 0] * sin(chi) * sin(eta)
            - cos(chi) * sin(eta) * (N_phi[0, 0] * cos(phi) + N_phi[1, 0] * sin(phi))
            - cos(eta) * (N_phi[0, 0] * sin(phi) - N_phi[1, 0] * cos(phi))
        )
        ks = atan2(A, B)
        mu = atan2(cos(theta) * cos(qaz), -sin(theta)) + ks
        yield mu, eta, chi, phi


def __calc_sample_con_eta_chi(
    samp_constraints: Dict[str, Optional[float]],
    qaz: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    eta = samp_constraints["eta"]
    chi = samp_constraints["chi"]

    A = N_phi[1, 0] * cos(chi) * cos(eta) - N_phi[0, 0] * sin(eta)
    B = N_phi[0, 0] * cos(chi) * cos(eta) + N_phi[1, 0] * sin(eta)
    if is_small(A) and is_small(B):
        raise DiffcalcException(
            "Phi cannot be chosen uniquely. Please choose a different set of constraints."
        )
    else:
        ks = atan2(A, B)
    try:
        acos_V00 = acos(
            bound(
                (cos(theta) * sin(qaz) - N_phi[2, 0] * cos(eta) * sin(chi))
                / sqrt(A**2 + B**2)
            )
        )
    except AssertionError:
        return
    if is_small(acos_V00):
        phi_list = [
            ks,
        ]
    else:
        phi_list = [acos_V00 + ks, -acos_V00 + ks]
    for phi in phi_list:
        A10 = (
            N_phi[0, 0] * cos(phi) * sin(chi)
            + N_phi[1, 0] * sin(chi) * sin(phi)
            - N_phi[2, 0] * cos(chi)
        )
        B10 = (
            -N_phi[2, 0] * sin(chi) * sin(eta)
            - (cos(chi) * cos(phi) * sin(eta) + cos(eta) * sin(phi)) * N_phi[0, 0]
            - (cos(chi) * sin(eta) * sin(phi) - cos(eta) * cos(phi)) * N_phi[1, 0]
        )
        V10 = -sin(theta)
        A20 = (
            -N_phi[2, 0] * sin(chi) * sin(eta)
            - (cos(chi) * cos(phi) * sin(eta) + cos(eta) * sin(phi)) * N_phi[0, 0]
            - (cos(chi) * sin(eta) * sin(phi) - cos(eta) * cos(phi)) * N_phi[1, 0]
        )
        B20 = (
            -N_phi[0, 0] * cos(phi) * sin(chi)
            - N_phi[1, 0] * sin(chi) * sin(phi)
            + N_phi[2, 0] * cos(chi)
        )
        V20 = cos(qaz) * cos(theta)
        sin_mu = (V10 * B20 - V20 * B10) * sign(A10 * B20 - A20 * B10)
        cos_mu = (V10 * A20 - V20 * A10) * sign(B10 * A20 - B20 * A10)
        if is_small(sin_mu) and is_small(cos_mu):
            raise DiffcalcException(
                "Mu cannot be chosen uniquely. Please choose a different set of constraints."
            )
        mu = atan2(sin_mu, cos_mu)
        yield mu, eta, chi, phi


def _calc_sample_con_two_sample_and_detector(
    samp_constraints: Dict[str, Optional[float]],
    qaz: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float]]:
    """Return sample angles.

    Available combinations:
    chi, phi, detector
    mu, eta, detector
    mu, phi, detector
    mu, chi, detector
    eta, phi, detector
    eta, chi, detector
    """
    if "mu" in samp_constraints and "eta" in samp_constraints:
        mu = samp_constraints["mu"]
        eta = samp_constraints["eta"]
        yield from __calc_sample_con_mu_eta(mu, eta, qaz, theta, N_phi)
    elif "omega" in samp_constraints and "bisect" in samp_constraints:
        yield from __calc_sample_con_omega_bisect(samp_constraints, qaz, theta, N_phi)
    elif "mu" in samp_constraints and "bisect" in samp_constraints:
        yield from __calc_sample_con_mu_bisect(samp_constraints, qaz, theta, N_phi)
    elif "eta" in samp_constraints and "bisect" in samp_constraints:
        yield from __calc_sample_con_eta_bisect(samp_constraints, qaz, theta, N_phi)
    elif "chi" in samp_constraints and "phi" in samp_constraints:
        yield from __calc_sample_con_chi_phi(samp_constraints, qaz, theta, N_phi)
    elif "mu" in samp_constraints and "phi" in samp_constraints:
        yield from __calc_sample_con_mu_phi(samp_constraints, qaz, theta, N_phi)
    elif "mu" in samp_constraints and "chi" in samp_constraints:
        yield from __calc_sample_con_mu_chi(samp_constraints, qaz, theta, N_phi)
    elif "eta" in samp_constraints and "phi" in samp_constraints:
        yield from __calc_sample_con_eta_phi(samp_constraints, qaz, theta, N_phi)
    elif "eta" in samp_constraints and "chi" in samp_constraints:
        yield from __calc_sample_con_eta_chi(samp_constraints, qaz, theta, N_phi)
    else:
        raise DiffcalcException(
            "No code yet to handle this combination of 2 sample "
            "constraints and one detector:" + str(samp_constraints)
        )
