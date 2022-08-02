"""Module implementing intermediate calculations in constrained reference geometry."""
from math import acos, asin, atan2, cos, pi, sin, sqrt
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
from diffcalc.hkl.geometry import rot_CHI, rot_PHI
from diffcalc.util import (
    DiffcalcException,
    bound,
    is_small,
    sign,
    x_rotation,
    z_rotation,
)
from numpy.linalg import inv


def __get_phi_and_qaz(
    chi: float, eta: float, mu: float, V: np.ndarray
) -> Tuple[float, float]:
    a = sin(chi) * cos(eta)
    b = sin(chi) * sin(eta) * sin(mu) - cos(chi) * cos(mu)
    sin_qaz = V[2, 0] * a - V[2, 2] * b
    cos_qaz = -V[2, 2] * a - V[2, 0] * b
    # atan2_xi = atan2(V[2, 2] * a + V[2, 0] * b,
    #           V[2, 0] * a - V[2, 2] * b)                        # (54)
    qaz = atan2(sin_qaz, cos_qaz)  # (54)

    a = sin(chi) * sin(mu) - cos(mu) * cos(chi) * sin(eta)
    b = cos(mu) * cos(eta)
    phi = atan2(V[1, 1] * a - V[0, 1] * b, V[0, 1] * a + V[1, 1] * b)  # (55)
    #        if is_small(mu+pi/2) and is_small(eta) and False:
    #            phi_general = phi
    #            # solved in extensions_to_yous_paper.wxm
    #            phi = atan2(V[1, 1], V[0, 1])
    #            logger.debug("phi = %.3f or %.3f (std)",
    #                        phi*TODEG, phi_general*TODEG )

    return qaz, phi


def __get_chi_and_qaz(mu: float, eta: float, V: np.ndarray) -> Tuple[float, float]:
    A = sin(mu)
    B = -cos(mu) * sin(eta)
    sin_chi = A * V[1, 0] + B * V[1, 2]
    cos_chi = B * V[1, 0] - A * V[1, 2]
    if is_small(sin_chi) and is_small(cos_chi):
        raise DiffcalcException(
            "Chi cannot be chosen uniquely. Please choose a different set of constraints."
        )
    chi = atan2(sin_chi, cos_chi)

    A = sin(eta)
    B = cos(eta) * sin(mu)
    sin_qaz = A * V[0, 1] + B * V[2, 1]
    cos_qaz = B * V[0, 1] - A * V[2, 1]
    qaz = atan2(sin_qaz, cos_qaz)
    return qaz, chi


def __calc_sample_ref_con_chi_phi(
    samp_constraints: Dict[str, Optional[float]],
    psi: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    chi = samp_constraints["chi"]
    phi = samp_constraints["phi"]

    CHI = rot_CHI(chi)
    PHI = rot_PHI(phi)
    THETA = z_rotation(-theta)
    PSI = x_rotation(psi)
    V = CHI @ PHI @ N_phi @ PSI.T @ THETA.T  # (46)

    # atan2_xi = atan2(-V[2, 0], V[2, 2])
    # atan2_eta = atan2(-V[0, 1], V[1, 1])
    # atan2_mu = atan2(-V[2, 1], sqrt(V[2, 2] ** 2 + V[2, 0] ** 2))
    try:
        asin_mu = asin(bound(-V[2, 1]))
    except AssertionError:
        return
    if is_small(cos(asin_mu)):
        mu_vals = [asin_mu]
    else:
        mu_vals = [asin_mu, pi - asin_mu]
    for mu in mu_vals:
        sgn_cosmu = sign(cos(mu))
        sin_qaz = sgn_cosmu * V[2, 2]
        cos_qaz = sgn_cosmu * V[2, 0]
        sin_eta = -sgn_cosmu * V[0, 1]
        cos_eta = sgn_cosmu * V[1, 1]
        if is_small(sin_eta) and is_small(cos_eta):
            raise DiffcalcException(
                "Position eta cannot be chosen uniquely. Please choose a different set of constraints."
            )
        if is_small(sin_qaz) and is_small(cos_qaz):
            raise DiffcalcException(
                "Scattering plane orientation qaz cannot be chosen uniquely. Please choose a different set of constraints."
            )
        # xi = atan2(-sgn_cosmu * V[2, 0], sgn_cosmu * V[2, 2])
        qaz = atan2(sin_qaz, cos_qaz)
        eta = atan2(sin_eta, cos_eta)
        yield qaz, psi, mu, eta, chi, phi


def __calc_sample_ref_con_mu_eta(
    samp_constraints: Dict[str, Optional[float]],
    psi: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    mu = samp_constraints["mu"]
    eta = samp_constraints["eta"]

    THETA = z_rotation(-theta)
    PSI = x_rotation(psi)
    V = N_phi @ PSI.T @ THETA.T  # (49)
    try:
        bot = bound(-V[2, 1] / sqrt(sin(eta) ** 2 * cos(mu) ** 2 + sin(mu) ** 2))
    except AssertionError:
        return
    if is_small(cos(mu) * sin(eta)):
        eps = atan2(sin(eta) * cos(mu), sin(mu))
        chi_vals = [eps + acos(bot), eps - acos(bot)]
    else:
        eps = atan2(sin(mu), sin(eta) * cos(mu))
        chi_vals = [asin(bot) - eps, pi - asin(bot) - eps]  # (52)

    ## Choose final chi solution here to obtain compatable xi and mu
    ## TODO: This temporary solution works only for one case used on i07
    ##       Return a list of possible solutions?
    # if is_small(eta) and is_small(mu + pi / 2):
    #    for chi in _generate_transformed_values(chi_orig):
    #        if  pi / 2 <= chi < pi:
    #            break
    # else:
    #    chi = chi_orig

    for chi in chi_vals:
        qaz, phi = __get_phi_and_qaz(chi, eta, mu, V)
        yield qaz, psi, mu, eta, chi, phi


def __calc_sample_ref_con_chi_eta(
    samp_constraints: Dict[str, Optional[float]],
    psi: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    chi = samp_constraints["chi"]
    eta = samp_constraints["eta"]

    THETA = z_rotation(-theta)
    PSI = x_rotation(psi)
    V = N_phi @ PSI.T @ THETA.T  # (49)
    try:
        bot = bound(-V[2, 1] / sqrt(sin(eta) ** 2 * sin(chi) ** 2 + cos(chi) ** 2))
    except AssertionError:
        return
    if is_small(cos(chi)):
        eps = atan2(cos(chi), sin(chi) * sin(eta))
        mu_vals = [eps + acos(bot), eps - acos(bot)]
    else:
        eps = atan2(sin(chi) * sin(eta), cos(chi))
        mu_vals = [asin(bot) - eps, pi - asin(bot) - eps]  # (52)

    for mu in mu_vals:
        qaz, phi = __get_phi_and_qaz(chi, eta, mu, V)
        yield qaz, psi, mu, eta, chi, phi


def __calc_sample_ref_con_chi_mu(
    samp_constraints: Dict[str, Optional[float]],
    psi: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    chi = samp_constraints["chi"]
    mu = samp_constraints["mu"]

    THETA = z_rotation(-theta)
    PSI = x_rotation(psi)
    V = N_phi @ PSI.T @ THETA.T  # (49)

    try:
        asin_eta = asin(bound((-V[2, 1] - cos(chi) * sin(mu)) / (sin(chi) * cos(mu))))
    except AssertionError:
        return

    for eta in [asin_eta, pi - asin_eta]:
        qaz, phi = __get_phi_and_qaz(chi, eta, mu, V)
        yield qaz, psi, mu, eta, chi, phi


def __calc_sample_ref_con_mu_phi(
    samp_constraints: Dict[str, Optional[float]],
    psi: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    mu = samp_constraints["mu"]
    phi = samp_constraints["phi"]

    PHI = rot_PHI(phi)
    THETA = z_rotation(-theta)
    PSI = x_rotation(psi)
    V = THETA @ PSI @ inv(N_phi) @ PHI.T

    if is_small(cos(mu)):
        raise DiffcalcException(
            "Eta cannot be chosen uniquely. Please choose a different set of constraints."
        )
    try:
        acos_eta = acos(bound(V[1, 1] / cos(mu)))
    except AssertionError:
        return
    for eta in [acos_eta, -acos_eta]:
        qaz, chi = __get_chi_and_qaz(mu, eta, V)
        yield qaz, psi, mu, eta, chi, phi


def __calc_sample_ref_con_eta_phi(
    samp_constraints: Dict[str, Optional[float]],
    psi: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    eta = samp_constraints["eta"]
    phi = samp_constraints["phi"]

    PHI = rot_PHI(phi)
    THETA = z_rotation(-theta)
    PSI = x_rotation(psi)
    V = THETA @ PSI @ inv(N_phi) @ PHI.T

    if is_small(cos(eta)):
        raise DiffcalcException(
            "Mu cannot be chosen uniquely. Please choose a different set of constraints."
        )
    try:
        acos_mu = acos(bound(V[1, 1] / cos(eta)))
    except AssertionError:
        return
    for mu in [acos_mu, -acos_mu]:
        qaz, chi = __get_chi_and_qaz(mu, eta, V)
        yield qaz, psi, mu, eta, chi, phi


def _calc_sample_con_two_sample_and_reference(
    samp_constraints: Dict[str, Optional[float]],
    psi: float,
    theta: float,
    N_phi: np.ndarray,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    """Return sample angles.

    Available combinations:
    chi, phi, reference
    mu, eta, reference,
    chi, eta, reference
    chi, mu, reference
    mu, phi, reference
    eta, phi, reference
    """
    if "chi" in samp_constraints and "phi" in samp_constraints:
        yield from __calc_sample_ref_con_chi_phi(samp_constraints, psi, theta, N_phi)
    elif "mu" in samp_constraints and "eta" in samp_constraints:
        yield from __calc_sample_ref_con_mu_eta(samp_constraints, psi, theta, N_phi)
    elif "chi" in samp_constraints and "eta" in samp_constraints:
        yield from __calc_sample_ref_con_chi_eta(samp_constraints, psi, theta, N_phi)
    elif "chi" in samp_constraints and "mu" in samp_constraints:
        yield from __calc_sample_ref_con_chi_mu(samp_constraints, psi, theta, N_phi)
    elif "mu" in samp_constraints and "phi" in samp_constraints:
        yield from __calc_sample_ref_con_mu_phi(samp_constraints, psi, theta, N_phi)
    elif "eta" in samp_constraints and "phi" in samp_constraints:
        yield from __calc_sample_ref_con_eta_phi(samp_constraints, psi, theta, N_phi)
    else:
        raise DiffcalcException(
            "No code yet to handle this combination of 2 sample "
            "constraints and one reference:" + str(samp_constraints)
        )
