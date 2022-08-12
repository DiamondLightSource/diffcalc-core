"""Module implementing intermediate calculations used in HKLCalculation class."""
import logging
from math import acos, asin, atan2, cos, degrees, sin, sqrt
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from diffcalc.hkl import calc_detector, calc_reference, calc_sample
from diffcalc.util import DiffcalcException, bound, is_small, normalised, sign


def _calc_remaining_reference_angles(
    name: str, value: float, theta: float, tau: float
) -> Tuple[float, float]:
    """Return alpha and beta given one of a_eq_b, alpha, beta or psi."""
    UNREACHABLE_MSG = (
        "The current combination of constraints with %s = %.4f\n"
        "prohibits a solution for the specified reflection."
    )
    try:
        if name == "psi":
            psi = value
            # Equation 26 for alpha
            sin_alpha = cos(tau) * sin(theta) - cos(theta) * sin(tau) * cos(psi)
            alpha = asin(bound(sin_alpha))
            # Equation 27 for beta
            sin_beta = cos(tau) * sin(theta) + cos(theta) * sin(tau) * cos(psi)
            beta = asin(bound(sin_beta))
        elif name == "a_eq_b" or name == "bin_eq_bout":
            alpha = beta = asin(bound(cos(tau) * sin(theta)))  # (24)
        elif name == "alpha" or name == "betain":
            alpha = value  # (24)
            sin_beta = 2 * sin(theta) * cos(tau) - sin(alpha)
            beta = asin(bound(sin_beta))
        elif name == "beta" or name == "betaout":
            beta = value
            sin_alpha = 2 * sin(theta) * cos(tau) - sin(beta)  # (24)
            alpha = asin(bound(sin_alpha))
        else:
            raise DiffcalcException(
                "Cannot calculate alpha and beta reference angles "
                f"using {name} constraint."
            )
        return alpha, beta
    except AssertionError:
        raise DiffcalcException(UNREACHABLE_MSG % (name, degrees(value)))


def _calc_det_sample_reference(
    det_constraint: Dict[str, Optional[float]],
    naz_constraint: Dict[str, Optional[float]],
    samp_constraints: Dict[str, Optional[float]],
    h_phi: np.ndarray,
    n_phi: np.ndarray,
    theta: float,
    alpha: float,
    tau: float,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    N_phi = calc_sample._calc_N(h_phi, n_phi)
    if len(samp_constraints) == 1:
        for (qaz, naz, delta, nu,) in calc_detector._calc_detector_con_det_or_naz(
            det_constraint, naz_constraint, theta, tau, alpha
        ):
            for (mu, eta, chi, phi,) in calc_sample._calc_remaining_sample_angles(
                samp_constraints, theta, alpha, qaz, naz, N_phi
            ):
                yield mu, delta, nu, eta, chi, phi

    elif len(samp_constraints) == 2:
        if det_constraint:
            _calc_remaining_detector_angles = {
                "delta": calc_detector._calc_remaining_detector_angles_delta,
                "nu": calc_detector._calc_remaining_detector_angles_nu,
                "qaz": calc_detector._calc_remaining_detector_angles_qaz,
            }
            det_constraint_name, det_constraint_val = next(iter(det_constraint.items()))
            for delta, nu, qaz in _calc_remaining_detector_angles[det_constraint_name](
                det_constraint_val, theta
            ):
                for (
                    mu,
                    eta,
                    chi,
                    phi,
                ) in calc_sample._calc_sample_con_two_sample_and_detector(
                    samp_constraints, qaz, theta, N_phi
                ):
                    yield mu, delta, nu, eta, chi, phi
        else:
            raise DiffcalcException(
                "No code yet to handle this combination of detector and sample constraints."
            )


def _calc_two_sample_and_reference(
    samp_constraints: Dict[str, Optional[float]],
    h_phi: np.ndarray,
    n_phi: np.ndarray,
    theta: float,
    psi: float,
) -> Iterator[Tuple[float, float, float, float, float, float]]:

    N_phi = calc_sample._calc_N(h_phi, n_phi)
    for angles in calc_reference._calc_sample_con_two_sample_and_reference(
        samp_constraints, psi, theta, N_phi
    ):
        qaz, psi, mu, eta, chi, phi = angles
        values_in_deg = tuple(degrees(v) for v in angles)
        logging.debug(
            "Initial angles: xi=%.3f, psi=%.3f, mu=%.3f, "
            "eta=%.3f, chi=%.3f, phi=%.3f" % values_in_deg
        )  # Try to find a solution for each possible transformed xi

        logging.debug("")
        msg = f"---Trying psi={degrees(psi):.3f}, qaz={degrees(qaz):.3f}"
        logging.debug(msg)

        for delta, nu, _ in calc_detector._calc_remaining_detector_angles_qaz(
            qaz, theta
        ):
            logging.debug("delta=%.3f, %s=%.3f", degrees(delta), "nu", degrees(nu))
            # for mu, eta, chi, phi in self._generate_sample_solutions(
            #    mu, eta, chi, phi, samp_constraints.keys(), delta,
            #    nu, wavelength, (h, k, l), ref_constraint_name,
            #    ref_constraint_value):
            yield mu, delta, nu, eta, chi, phi


def __get_qaz_value(
    mu: float,
    eta: float,
    chi: float,
    phi: float,
    h_phi: np.ndarray,
    theta: float,
) -> float:
    h_phi_norm = normalised(h_phi)  # (68,69)
    h0, h1, h2 = h_phi_norm[0, 0], h_phi_norm[1, 0], h_phi_norm[2, 0]

    V0 = (
        h2 * cos(eta) * sin(chi)
        + (h0 * cos(chi) * cos(eta) + h1 * sin(eta)) * cos(phi)
        + (h1 * cos(chi) * cos(eta) - h0 * sin(eta)) * sin(phi)
    )
    V2 = (
        -h2 * sin(chi) * sin(eta) * sin(mu)
        + h2 * cos(chi) * cos(mu)
        - (
            h0 * cos(mu) * sin(chi)
            + (h0 * cos(chi) * sin(eta) - h1 * cos(eta)) * sin(mu)
        )
        * cos(phi)
        - (
            h1 * cos(mu) * sin(chi)
            + (h1 * cos(chi) * sin(eta) + h0 * cos(eta)) * sin(mu)
        )
        * sin(phi)
    )
    sgn_theta = sign(cos(theta))
    qaz = atan2(sgn_theta * V0, sgn_theta * V2)
    return qaz


def __get_last_sample_angle(
    samp_constraints: Dict[str, Optional[float]],
    h_phi: np.ndarray,
    theta: float,
) -> List[float]:
    h_phi_norm = normalised(h_phi)  # (68,69)
    h0, h1, h2 = h_phi_norm[0, 0], h_phi_norm[1, 0], h_phi_norm[2, 0]

    if "mu" not in samp_constraints:
        eta = samp_constraints["eta"]
        chi = samp_constraints["chi"]
        phi = samp_constraints["phi"]

        A = h0 * cos(phi) * sin(chi) + h1 * sin(chi) * sin(phi) - h2 * cos(chi)
        B = (
            -h2 * sin(chi) * sin(eta)
            - (h0 * cos(chi) * sin(eta) - h1 * cos(eta)) * cos(phi)
            - (h1 * cos(chi) * sin(eta) + h0 * cos(eta)) * sin(phi)
        )
        C = -sin(theta)
    elif "eta" not in samp_constraints:
        mu = samp_constraints["mu"]
        chi = samp_constraints["chi"]
        phi = samp_constraints["phi"]

        A = (
            -h0 * cos(chi) * cos(mu) * cos(phi)
            - h1 * cos(chi) * cos(mu) * sin(phi)
            - h2 * cos(mu) * sin(chi)
        )
        B = h1 * cos(mu) * cos(phi) - h0 * cos(mu) * sin(phi)
        C = (
            -h0 * cos(phi) * sin(chi) * sin(mu)
            - h1 * sin(chi) * sin(mu) * sin(phi)
            + h2 * cos(chi) * sin(mu)
            - sin(theta)
        )
    elif "chi" not in samp_constraints:
        mu = samp_constraints["mu"]
        eta = samp_constraints["eta"]
        phi = samp_constraints["phi"]

        A = -h2 * cos(mu) * sin(eta) + h0 * cos(phi) * sin(mu) + h1 * sin(mu) * sin(phi)
        B = (
            -h0 * cos(mu) * cos(phi) * sin(eta)
            - h1 * cos(mu) * sin(eta) * sin(phi)
            - h2 * sin(mu)
        )
        C = (
            -h1 * cos(eta) * cos(mu) * cos(phi)
            + h0 * cos(eta) * cos(mu) * sin(phi)
            - sin(theta)
        )
    elif "phi" not in samp_constraints:
        mu = samp_constraints["mu"]
        eta = samp_constraints["eta"]
        chi = samp_constraints["chi"]

        A = h1 * sin(chi) * sin(mu) - (h1 * cos(chi) * sin(eta) + h0 * cos(eta)) * cos(
            mu
        )
        B = h0 * sin(chi) * sin(mu) - (h0 * cos(chi) * sin(eta) - h1 * cos(eta)) * cos(
            mu
        )
        C = h2 * cos(mu) * sin(chi) * sin(eta) + h2 * cos(chi) * sin(mu) - sin(theta)

    if is_small(A) and is_small(B):
        raise DiffcalcException(
            "Sample orientation cannot be chosen uniquely.\n"
            "Please choose a different set of constraints."
        )
    ks = atan2(A, B)
    acos_alp = acos(bound(C / sqrt(A**2 + B**2)))
    if is_small(acos_alp):
        alp_list = [
            ks,
        ]
    else:
        alp_list = [acos_alp + ks, -acos_alp + ks]
    return alp_list


def __calc_eta_chi_phi(
    samp_constraints: Dict[str, Optional[float]],
    h_phi: np.ndarray,
    theta: float,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    eta = samp_constraints["eta"]
    chi = samp_constraints["chi"]
    phi = samp_constraints["phi"]

    try:
        mu_vals = __get_last_sample_angle(samp_constraints, h_phi, theta)
    except AssertionError:
        return
    for mu in mu_vals:
        qaz = __get_qaz_value(mu, eta, chi, phi, h_phi, theta)
        logging.debug("--- Trying mu:%.f qaz_%.f", degrees(mu), degrees(qaz))
        for delta, nu, _ in calc_detector._calc_remaining_detector_angles_qaz(
            qaz, theta
        ):
            logging.debug("delta=%.3f, %s=%.3f", degrees(delta), "nu", degrees(nu))
            yield mu, delta, nu, eta, chi, phi


def __calc_mu_chi_phi(
    samp_constraints: Dict[str, Optional[float]],
    h_phi: np.ndarray,
    theta: float,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    mu = samp_constraints["mu"]
    chi = samp_constraints["chi"]
    phi = samp_constraints["phi"]

    try:
        eta_vals = __get_last_sample_angle(samp_constraints, h_phi, theta)
    except AssertionError:
        return
    for eta in eta_vals:
        qaz = __get_qaz_value(mu, eta, chi, phi, h_phi, theta)
        logging.debug("--- Trying eta:%.f qaz_%.f", degrees(eta), degrees(qaz))
        for delta, nu, _ in calc_detector._calc_remaining_detector_angles_qaz(
            qaz, theta
        ):
            logging.debug("delta=%.3f, %s=%.3f", degrees(delta), "nu", degrees(nu))
            yield mu, delta, nu, eta, chi, phi


def __calc_mu_eta_phi(
    samp_constraints: Dict[str, Optional[float]],
    h_phi: np.ndarray,
    theta: float,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    mu = samp_constraints["mu"]
    eta = samp_constraints["eta"]
    phi = samp_constraints["phi"]

    try:
        chi_vals = __get_last_sample_angle(samp_constraints, h_phi, theta)
    except AssertionError:
        return
    for chi in chi_vals:
        qaz = __get_qaz_value(mu, eta, chi, phi, h_phi, theta)
        logging.debug("--- Trying chi:%.f qaz_%.f", degrees(chi), degrees(qaz))
        for delta, nu, _ in calc_detector._calc_remaining_detector_angles_qaz(
            qaz, theta
        ):
            logging.debug("delta=%.3f, nu=%.3f", degrees(delta), degrees(nu))
            yield mu, delta, nu, eta, chi, phi


def __calc_mu_eta_chi(
    samp_constraints: Dict[str, Optional[float]],
    h_phi: np.ndarray,
    theta: float,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    mu = samp_constraints["mu"]
    eta = samp_constraints["eta"]
    chi = samp_constraints["chi"]

    try:
        phi_vals = __get_last_sample_angle(samp_constraints, h_phi, theta)
    except AssertionError:
        return
    for phi in phi_vals:
        qaz = __get_qaz_value(mu, eta, chi, phi, h_phi, theta)
        logging.debug("--- Trying phi:%.f qaz_%.f", degrees(phi), degrees(qaz))
        for delta, nu, _ in calc_detector._calc_remaining_detector_angles_qaz(
            qaz, theta
        ):
            logging.debug("delta=%.3f, nu=%.3f", degrees(delta), degrees(nu))
            yield mu, delta, nu, eta, chi, phi


def _calc_three_sample(
    samp_constraints: Dict[str, Optional[float]],
    h_phi: np.ndarray,
    theta: float,
) -> Iterator[Tuple[float, float, float, float, float, float]]:
    if "mu" not in samp_constraints:
        yield from __calc_eta_chi_phi(samp_constraints, h_phi, theta)
    elif "eta" not in samp_constraints:
        yield from __calc_mu_chi_phi(samp_constraints, h_phi, theta)
    elif "chi" not in samp_constraints:
        yield from __calc_mu_eta_phi(samp_constraints, h_phi, theta)
    elif "phi" not in samp_constraints:
        yield from __calc_mu_eta_chi(samp_constraints, h_phi, theta)
    else:
        raise DiffcalcException("Internal error: Invalid set of sample constraints.")
