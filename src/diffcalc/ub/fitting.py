"""Fit crystal lattice and U matrix to reflection data.

A module implementing fitting routines for refining crystal lattice parameters
and U matrix using reflection data.
"""
from math import atan2, cos, pi, sin, sqrt
from typing import List, Sequence, Tuple

import numpy as np
from diffcalc.hkl.geometry import Position, get_rotation_matrices
from diffcalc.ub.crystal import Crystal
from diffcalc.ub.reference import Reflection
from diffcalc.util import DiffcalcException, I, angle_between_vectors, sign
from numpy.linalg import inv, norm
from scipy.optimize import minimize


def _get_refl_hkl(
    refl_list: List[Reflection],
) -> List[Tuple[np.ndarray, Position, float]]:
    refl_data = []
    for refl in refl_list:
        refl_data.append(
            (np.array([[refl.h], [refl.k], [refl.l]]), refl.pos, refl.energy)
        )
    return refl_data


def _func_crystal(
    vals: Sequence[float], uc_system: str, refl_data: Tuple[np.ndarray, Position, float]
) -> float:
    try:
        trial_cr = Crystal("trial", uc_system, *vals)
    except Exception:
        return 1e6

    res = 0
    for (hkl_vals, pos_vals, en) in refl_data:
        wl = 12.3984 / en
        [_, DELTA, NU, _, _, _] = get_rotation_matrices(pos_vals)
        q_pos = (NU @ DELTA - I) @ np.array([[0], [2 * pi / wl], [0]])
        q_hkl = trial_cr.B @ hkl_vals
        res += (norm(q_pos) - norm(q_hkl)) ** 2
    return res


def _func_orient(
    vals, crystal: Crystal, refl_data: Tuple[np.ndarray, Position, float]
) -> float:
    quat = _get_quat_from_u123(*vals)
    trial_u = _get_rot_matrix(*quat)
    tmp_ub = trial_u @ crystal.B

    res = 0.0
    for (hkl_vals, pos_vals, en) in refl_data:
        wl = 12.3984 / en
        [MU, DELTA, NU, ETA, CHI, PHI] = get_rotation_matrices(pos_vals)
        q_del = (NU @ DELTA - I) @ np.array([[0], [2 * pi / wl], [0]])
        q_vals = inv(PHI) @ inv(CHI) @ inv(ETA) @ inv(MU) @ q_del

        q_hkl = tmp_ub @ hkl_vals
        res += angle_between_vectors(q_hkl, q_vals)
    return res


def _get_rot_matrix(q0: float, q1: float, q2: float, q3: float) -> np.ndarray:
    rot = np.array(
        [
            [
                q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2,
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ],
            [
                2.0 * (q1 * q2 + q0 * q3),
                q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2,
                2.0 * (q2 * q3 - q0 * q1),
            ],
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2,
            ],
        ]
    )
    return rot


def _get_init_u123(um: np.ndarray) -> Tuple[float, float, float]:

    tr = um[0, 0] + um[1, 1] + um[2, 2]
    sgn_q1 = sign(um[2, 1] - um[1, 2])
    sgn_q2 = sign(um[0, 2] - um[2, 0])
    sgn_q3 = sign(um[1, 0] - um[0, 1])
    q0 = sqrt(1.0 + tr) / 2.0
    q1 = sgn_q1 * sqrt(1.0 + um[0, 0] - um[1, 1] - um[2, 2]) / 2.0
    q2 = sgn_q2 * sqrt(1.0 - um[0, 0] + um[1, 1] - um[2, 2]) / 2.0
    q3 = sgn_q3 * sqrt(1.0 - um[0, 0] - um[1, 1] + um[2, 2]) / 2.0
    u1 = (1.0 - um[0, 0]) / 2.0
    u2 = atan2(q0, q1) / (2.0 * pi)
    u3 = atan2(q2, q3) / (2.0 * pi)
    if u2 < 0:
        u2 += 1.0
    if u3 < 0:
        u3 += 1.0
    return u1, u2, u3


def _get_quat_from_u123(
    u1: float, u2: float, u3: float
) -> Tuple[float, float, float, float]:
    q0, q1 = sqrt(1.0 - u1) * sin(2.0 * pi * u2), sqrt(1.0 - u1) * cos(2.0 * pi * u2)
    q2, q3 = sqrt(u1) * sin(2.0 * pi * u3), sqrt(u1) * cos(2.0 * pi * u3)
    return q0, q1, q2, q3


def _get_uc_upper_limits(system: str) -> List[float]:
    max_unit = 100.0
    if system == "Triclinic":
        return [max_unit, max_unit, max_unit, 180.0, 180.0, 180.0]
    elif system == "Monoclinic":
        return [max_unit, max_unit, max_unit, 180.0]
    elif system == "Orthorhombic":
        return [
            max_unit,
            max_unit,
            max_unit,
        ]
    elif system == "Tetragonal" or system == "Hexagonal":
        return [
            max_unit,
            max_unit,
        ]
    elif system == "Rhombohedral":
        return [
            max_unit,
            180.0,
        ]
    elif system == "Cubic":
        return [
            max_unit,
        ]
    else:
        raise TypeError("Invalid crystal system parameter: %s" % str(system))


def fit_crystal(crystal: Crystal, refl_list: List[Reflection]) -> Crystal:
    """Fit crystal lattice parameters to reference reflections.

    Parameters
    ----------
    crystal: Crystal
        Object containing initial crysta lattice parameter estimates
    refl_list: List[Reflection]
        List or reference reflection objects

    Returns
    -------
    Crystal
        Object containing refined crystal lattice parameters.

    Raises
    ------
    DiffcalcException
        If crystal lattice object not initialised.
    """
    try:
        xtal_system, xtal_params = crystal.get_lattice_params()
        start = xtal_params
        lower = [
            0,
        ] * len(xtal_params)
        upper = _get_uc_upper_limits(xtal_system)
    except AttributeError:
        raise DiffcalcException(
            "UB matrix not initialised. Cannot run UB matrix fitting procedure."
        )

    refl_data = _get_refl_hkl(refl_list)
    bounds = list(zip(lower, upper))
    res = minimize(
        _func_crystal,
        start,
        args=(xtal_system, refl_data),
        method="SLSQP",
        tol=1e-10,
        options={"disp": False, "maxiter": 10000, "eps": 1e-6, "ftol": 1e-10},
        bounds=bounds,
    )
    vals = res.x

    res_cr = Crystal("trial", xtal_system, *vals)
    # res_cr._set_cell_for_system(uc_system, *vals)
    return res_cr


def fit_u_matrix(
    init_u: np.ndarray, crystal: Crystal, refl_list: List[Reflection]
) -> np.ndarray:
    """Fit crystal lattice parameters to reference reflections.

    Parameters
    ----------
    init_u: np.ndarray
        Initial U matrix as (3, 3) NumPy array
    crystal: Crystal
        Object containing initial crystal lattice parameter estimates
    refl_list: List[Reflection]
        List or reference reflection objects

    Returns
    -------
    np.ndarray
        NumPy array with refined U matrix elements.

    Raises
    ------
    DiffcalcException
        If U matrix is not initialised.
    """
    try:
        start = list(_get_init_u123(init_u))
        lower = [0, 0, 0]
        upper = [1, 1, 1]
    except TypeError:
        raise DiffcalcException(
            "UB matrix not initialised. Cannot run UB matrix fitting procedure."
        )

    ref_data = _get_refl_hkl(refl_list)
    bounds = list(zip(lower, upper))
    res = minimize(
        _func_orient,
        start,
        args=(crystal, ref_data),
        method="SLSQP",
        tol=1e-10,
        options={"disp": False, "maxiter": 10000, "eps": 1e-6, "ftol": 1e-10},
        bounds=bounds,
    )
    vals = res.x

    q0, q1, q2, q3 = _get_quat_from_u123(*vals)
    res_u = _get_rot_matrix(q0, q1, q2, q3)
    # angle = 2. * acos(q0)
    # xr = q1 / sqrt(1. - q0 * q0)
    # yr = q2 / sqrt(1. - q0 * q0)
    # zr = q3 / sqrt(1. - q0 * q0)
    # print angle * TODEG, (xr, yr, zr), res
    return res_u
