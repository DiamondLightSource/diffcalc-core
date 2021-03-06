"""Routines for calculating miller indices and diffractometer positions.

Module implementing calculations based on UB matrix data and diffractometer
constraints.
"""
from copy import copy
from math import acos, asin, atan2, cos, degrees, isnan, pi, sin
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from diffcalc.hkl import calc_func
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position, get_rotation_matrices
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import (
    DiffcalcException,
    I,
    angle_between_vectors,
    bound,
    is_small,
    radians_equivalent,
    sign,
)
from numpy.linalg import inv


class HklCalculation:
    """Class for converting between miller indices and diffractometer position.

    Attributes
    ----------
    ubcalc: UBcalculation
        Reference to UBcalculation object containing UB matrix data.
    constraints:
        Reference to Constraints object containing diffractometer constraint settings.

    Methods
    -------
    get_hkl(pos: Position, wavelength: float) -> Tuple[float, float, float]
        Calculate miller indices corresponding to a diffractometer positions.
    get_virtual_angles(pos: Position, asdegrees: bool = True) -> Dict[str,float]
        Calculate pseudo-angles corresponding to a diffractometer position.
    """

    def __init__(self, ubcalc, constraints):
        self.ubcalc = ubcalc  # to get the UBMatrix
        self.constraints = constraints

    def __str__(self):
        """Return string representing class instance.

        Returns
        -------
        str:
            String representation of constraints table.
        """
        return self.constraints.__str__()

    def __repr_mode(self):
        return repr(self.constraints.asdict)

    def get_hkl(self, pos: Position, wavelength: float) -> Tuple[float, float, float]:
        """Calculate miller indices corresponding to a diffractometer positions.

        Parameters
        ----------
        pos: Position
            Diffractometer position.

        Returns
        -------
        Tuple[float, float, float]
            Miller indices corresponding to the specified diffractometer
            position at the given wavelength.
        """
        pos_in_rad = Position.asradians(pos)
        [MU, DELTA, NU, ETA, CHI, PHI] = get_rotation_matrices(pos_in_rad)

        q_lab = (NU @ DELTA - I) @ np.array([[0], [2 * pi / wavelength], [0]])  # 12

        hkl = inv(self.ubcalc.UB) @ inv(PHI) @ inv(CHI) @ inv(ETA) @ inv(MU) @ q_lab

        return hkl[0, 0], hkl[1, 0], hkl[2, 0]

    def get_virtual_angles(
        self, pos: Position, asdegrees: bool = True
    ) -> Dict[str, float]:
        """Calculate pseudo-angles corresponding to a diffractometer position.

        Parameters
        ----------
        pos: Position
            Diffractometer position.
        asdegrees: bool = True
            If True, return angles in degrees.

        Returns
        -------
        Dict[str, float]
            Returns alpha, beta, betain, betaout, naz, psi, qaz, tau, theta and
            ttheta angles.
        """
        pos_in_rad = Position.asradians(pos)
        theta, qaz = self._theta_and_qaz_from_detector_angles(
            pos_in_rad.delta, pos_in_rad.nu
        )  # (19)

        [MU, DELTA, NU, ETA, CHI, PHI] = get_rotation_matrices(pos_in_rad)
        Z = MU @ ETA @ CHI @ PHI
        D = NU @ DELTA

        # Compute incidence and outgoing angles bin and betaout
        surf_nphi = Z @ self.ubcalc.surf_nphi
        kin = np.array([[0], [1], [0]])
        kout = D @ np.array([[0], [1], [0]])
        betain = angle_between_vectors(kin, surf_nphi) - pi / 2.0
        betaout = pi / 2.0 - angle_between_vectors(kout, surf_nphi)

        n_lab = Z @ self.ubcalc.n_phi
        alpha = asin(bound(-n_lab[1, 0]))
        naz = atan2(n_lab[0, 0], n_lab[2, 0])  # (20)

        cos_tau = cos(alpha) * cos(theta) * cos(naz - qaz) + sin(alpha) * sin(theta)
        tau = acos(bound(cos_tau))  # (23)

        # Compute Tau using the dot product directly (THIS ALSO WORKS)
        # q_lab = ( (NU @ DELTA - I ) @ np.array([[0],[1],[0]])
        # norm = norm(q_lab)
        # q_lab = np.array([[1],[0],[0]]) if norm == 0 else q_lab * (1/norm)
        # tau_from_dot_product = acos(bound(dot3(q_lab, n_lab)))

        sin_beta = 2 * sin(theta) * cos(tau) - sin(alpha)
        beta = asin(bound(sin_beta))  # (24)

        psi = next(calc_func._calc_psi(alpha, theta, tau, qaz, naz))

        result = {
            "theta": theta,
            "ttheta": 2 * theta,
            "qaz": qaz,
            "alpha": alpha,
            "naz": naz,
            "tau": tau,
            "psi": psi,
            "beta": beta,
            "betain": betain,
            "betaout": betaout,
        }
        if asdegrees:
            result = {key: degrees(val) for key, val in result.items()}
        return result

    def get_position(
        self, h: float, k: float, l: float, wavelength: float, asdegrees: bool = True
    ) -> List[Tuple[Position, Dict[str, float]]]:
        """Calculate diffractometer position from miller indices and wavelength.

        The calculated positions and angles are verified by checking that they
        map to the requested miller indices.

        Parameters
        ----------
            h: float
                h miller index.
            k: float
                k miller index.
            l: float
                l miller index.
            wavelength: float
                wavelength in Angstroms.
            asdegrees: bool
                If True, return angles in degrees.

        Returns
        -------
            List[Tuple[Position, Dict[str, float]]]
                List of all solutions matching the input miller indices that
                consists of pairs of diffractometer position object and virtual
                angles dictionary.
        """
        pos_virtual_angles_pairs = self._calc_hkl_to_position(h, k, l, wavelength)
        assert pos_virtual_angles_pairs
        results = []

        for pos, virtual_angles in pos_virtual_angles_pairs:
            self._verify_pos_map_to_hkl(h, k, l, wavelength, pos)
            self._verify_virtual_angles(h, k, l, pos, virtual_angles)
            if asdegrees:
                res_pos = Position.asdegrees(pos)
                res_virtual_angles = {
                    key: degrees(val) for key, val in virtual_angles.items()
                }
            else:
                res_pos = Position.asradians(pos)
                res_virtual_angles = copy(virtual_angles)
            results.append((res_pos, res_virtual_angles))

        return results

    def _calc_nphi_alpha_tau(
        self,
        ref_constraint: Dict[str, Optional[float]],
        h_phi: np.ndarray,
        n_phi: np.ndarray,
        theta: float,
    ) -> Tuple[np.ndarray, float, float]:
        tau = angle_between_vectors(h_phi, self.ubcalc.n_phi)
        surf_tau = angle_between_vectors(h_phi, self.ubcalc.surf_nphi)

        ref_constraint_name, ref_constraint_value = next(iter(ref_constraint.items()))
        if is_small(sin(tau)):
            if ref_constraint_name == "psi":
                raise DiffcalcException(
                    "Azimuthal angle 'psi' is undefined as reference and scattering vectors parallel.\n"
                    "Please constrain one of the sample angles or choose different reference vector orientation."
                )
            elif ref_constraint_name == "a_eq_b":
                raise DiffcalcException(
                    "Reference constraint 'a_eq_b' is redundant as reference and scattering vectors are parallel.\n"
                    "Please constrain one of the sample angles or choose different reference vector orientation."
                )
        if is_small(sin(surf_tau)) and ref_constraint_name == "bin_eq_bout":
            raise DiffcalcException(
                "Reference constraint 'bin_eq_bout' is redundant as scattering vectors is parallel to the surface normal.\n"
                "Please select another constrain to define sample azimuthal orientation."
            )

        ### Reference constraint column ###

        if {"psi", "a_eq_b", "alpha", "beta"}.issuperset(ref_constraint.keys()):
            # An angle for the reference vector (n) is given      (Section 5.2)
            alpha, _ = calc_func._calc_remaining_reference_angles(
                ref_constraint_name, ref_constraint_value, theta, tau
            )
        elif {"bin_eq_bout", "betain", "betaout"}.issuperset(ref_constraint.keys()):
            alpha, _ = calc_func._calc_remaining_reference_angles(
                ref_constraint_name, ref_constraint_value, theta, surf_tau
            )
            tau = surf_tau
            n_phi = self.ubcalc.surf_nphi
        return n_phi, alpha, tau

    def _calc_hkl_to_position(
        self, h: float, k: float, l: float, wavelength: float
    ) -> List[Tuple[Position, Dict[str, float]]]:
        if not self.constraints.is_fully_constrained():
            raise DiffcalcException("Diffcalc is not fully constrained.")

        if not self.constraints.is_current_mode_implemented():
            raise DiffcalcException(
                "Sorry, the selected constraint combination is valid but "
                "is not implemented."
            )

        # constraints are dictionaries
        ref_constraint = self.constraints._reference
        det_constraint = self.constraints._detector
        naz_constraint = {"naz": self.constraints.naz} if self.constraints.naz else None
        samp_constraints = self.constraints._sample

        if "naz" in det_constraint:
            det_constraint.pop("naz")

        assert not (
            det_constraint and naz_constraint
        ), "Two 'detector' constraints given."

        h_phi = self.ubcalc.UB @ np.array([[h], [k], [l]])
        n_phi = self.ubcalc.n_phi
        theta = (
            self.ubcalc.get_ttheta_from_hkl((h, k, l), 12.39842 / wavelength) / 2.0
        )  # __calc_theta(h_phi, wavelength)
        alpha = None
        tau = None

        if ref_constraint:
            n_phi, alpha, tau = self._calc_nphi_alpha_tau(
                ref_constraint, h_phi, n_phi, theta
            )

        solution_tuples: List[Tuple[float, float, float, float, float, float]] = []
        if det_constraint or naz_constraint:

            solution_tuples.extend(
                calc_func._calc_given_det_sample_reference(
                    det_constraint,
                    naz_constraint,
                    samp_constraints,
                    h_phi,
                    n_phi,
                    theta,
                    alpha,
                    tau,
                )
            )

        elif len(samp_constraints) == 2:
            solution_tuples.extend(
                calc_func._calc_sample_given_two_sample_and_reference(
                    ref_constraint,
                    samp_constraints,
                    h_phi,
                    n_phi,
                    theta,
                    alpha,
                    tau,
                )
            )

        elif len(samp_constraints) == 3:
            solution_tuples.extend(
                calc_func._calc_angles_given_three_sample_constraints(
                    samp_constraints,
                    h_phi,
                    theta,
                )
            )

        if not solution_tuples:
            raise DiffcalcException(
                "No solutions were found. "
                "Please consider using an alternative set of constraints."
            )

        tidy_solutions = [
            self._tidy_degenerate_solutions(Position(*pos, False))
            for pos in solution_tuples
        ]

        # def _find_duplicate_angles(el):
        #    idx, tpl = el
        #    for tmp_tpl in filtered_solutions[idx:]:
        #        if False not in [abs(x-y) < SMALL for x,y in zip(tmp_tpl, tpl)]:
        #            return False
        #    return True
        # merged_solution_tuples = filter(_find_duplicate_angles, enumerate(filtered_solutions, 1))
        position_pseudo_angles_pairs = self._create_position_pseudo_angles_pairs(
            tidy_solutions
        )
        if not position_pseudo_angles_pairs:
            raise DiffcalcException(
                "No solutions were found. Please consider using "
                "an alternative pseudo-angle constraints."
            )

        return position_pseudo_angles_pairs

    def _create_position_pseudo_angles_pairs(
        self, merged_solution_tuples: List[Position]
    ) -> List[Tuple[Position, Dict[str, float]]]:

        position_pseudo_angles_pairs = []
        for position in merged_solution_tuples:
            # Create position
            # position = self._tidy_degenerate_solutions(position)
            # if position.phi <= -pi + SMALL:
            #    position.phi += 2 * pi
            # pseudo angles calculated along the way were for the initial solution
            # and may be invalid for the chosen solution TODO: anglesToHkl need no
            # longer check the pseudo_angles as they will be generated with the
            # same function and it will prove nothing
            pseudo_angles = self.get_virtual_angles(position, False)
            try:
                for constraint in [
                    self.constraints._reference,
                    self.constraints._detector,
                ]:
                    for constraint_name, constraint_value in constraint.items():
                        if constraint_name == "a_eq_b":
                            assert radians_equivalent(
                                pseudo_angles["alpha"], pseudo_angles["beta"]
                            )
                        elif constraint_name == "bin_eq_bout":
                            assert radians_equivalent(
                                pseudo_angles["betain"], pseudo_angles["betaout"]
                            )
                        elif constraint_name not in pseudo_angles:
                            continue
                        else:
                            assert radians_equivalent(
                                constraint_value, pseudo_angles[constraint_name]
                            )
                position_pseudo_angles_pairs.append((position, pseudo_angles))
            except AssertionError:
                continue
        return position_pseudo_angles_pairs

    def _tidy_degenerate_solutions(
        self, pos: Position, print_degenerate: bool = False
    ) -> Position:

        detector_like_constraint = self.constraints._detector or self.constraints.naz
        nu_constrained_to_0 = is_small(pos.nu) and detector_like_constraint
        mu_constrained_to_0 = is_small(pos.mu) and "mu" in self.constraints._sample
        delta_constrained_to_0 = is_small(pos.delta) and detector_like_constraint
        eta_constrained_to_0 = is_small(pos.eta) and "eta" in self.constraints._sample
        phi_not_constrained = "phi" not in self.constraints._sample

        if (
            nu_constrained_to_0
            and mu_constrained_to_0
            and phi_not_constrained
            and is_small(pos.chi)
        ):
            # constrained to vertical 4-circle like mode
            # phi || eta
            desired_eta = pos.delta / 2.0
            eta_diff = desired_eta - pos.eta
            if print_degenerate:
                print(
                    "DEGENERATE: with chi=0, phi and eta are colinear:"
                    "choosing eta = delta/2 by adding % 7.3f to eta and "
                    "removing it from phi. (mu=nu=0 only)" % degrees(eta_diff)
                )
                print("            original:", pos)
            newpos = Position(
                pos.mu,
                pos.delta,
                pos.nu,
                desired_eta,
                pos.chi,
                pos.phi - eta_diff,
                False,
            )

        elif (
            delta_constrained_to_0
            and eta_constrained_to_0
            and phi_not_constrained
            and is_small(pos.chi - pi / 2)
        ):
            # constrained to horizontal 4-circle like mode
            # phi || mu
            desired_mu = pos.nu / 2.0
            mu_diff = desired_mu - pos.mu
            if print_degenerate:
                print(
                    "DEGENERATE: with chi=90, phi and mu are colinear: choosing"
                    " mu = nu/2 by adding % 7.3f to mu and to phi. "
                    "(delta=eta=0 only)" % degrees(mu_diff)
                )
                print("            original:", pos)
            newpos = Position(
                desired_mu,
                pos.delta,
                pos.nu,
                pos.eta,
                pos.chi,
                pos.phi + mu_diff,
                False,
            )
        else:
            newpos = pos
        return newpos

    def _theta_and_qaz_from_detector_angles(
        self, delta: float, nu: float
    ) -> Tuple[float, float]:
        # Equation 19:
        cos_2theta = cos(delta) * cos(nu)
        theta = acos(cos_2theta) / 2.0
        sgn = sign(sin(2.0 * theta))
        qaz = atan2(sgn * sin(delta), sgn * cos(delta) * sin(nu))
        return theta, qaz

    def _verify_pos_map_to_hkl(
        self, h: float, k: float, l: float, wavelength: float, pos: Position
    ) -> None:
        hkl = self.get_hkl(pos, wavelength)
        e = 0.001
        if (abs(hkl[0] - h) > e) or (abs(hkl[1] - k) > e) or (abs(hkl[2] - l) > e):
            s = "ERROR: The angles calculated for hkl=({:f},{:f},{:f}) were {}.\n".format(
                h,
                k,
                l,
                str(pos),
            )
            s += "Converting these angles back to hkl resulted in hkl=" "(%f,%f,%f)" % (
                hkl[0],
                hkl[1],
                hkl[2],
            )
            raise DiffcalcException(s)

    def _verify_virtual_angles(
        self,
        h: float,
        k: float,
        l: float,
        pos: Position,
        virtual_angles: Dict[str, float],
    ) -> None:
        # Check that the virtual angles calculated/fixed during the hklToAngles
        # those read back from pos using anglesToVirtualAngles
        virtual_angles_readback = self.get_virtual_angles(pos, False)
        for key, val in virtual_angles.items():
            if val is not None:  # Some values calculated in some mode_selector
                r = virtual_angles_readback[key]
                if (not isnan(val) or not isnan(r)) and not radians_equivalent(
                    val, r, 1e-5
                ):
                    s = (
                        "ERROR: The angles calculated for hkl=(%f,%f,%f) with"
                        " mode=%s were %s.\n" % (h, k, l, self.__repr_mode(), str(pos))
                    )
                    s += (
                        "During verification the virtual angle %s resulting "
                        "from (or set for) this calculation of %f " % (key, val)
                    )
                    s += (
                        "did not match that calculated by "
                        "anglesToVirtualAngles of %f" % virtual_angles_readback[key]
                    )
                    raise DiffcalcException(s)

    @property
    def asdict(self) -> Dict[str, Any]:
        """Serialise the object into a JSON compatible dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing properties of hkl class. Can
            be unpacked to recreate HklCalculation object using fromdict
            class method below.

        """
        return {"ubcalc": self.ubcalc.asdict, "constraints": self.constraints.asdict}

    @classmethod
    def fromdict(cls, data: Dict[str, Any]) -> "HklCalculation":
        """Construct HklCalculation instance from a JSON compatible dictionary.

        Parameters
        ----------
        data: Dict[str, Any]
            Dictionary containing properties of hkl class, must have the equivalent
            structure of asdict method above.

        Returns
        -------
        HklCalculation
            Instance of this class created from the dictionary.

        """
        constraint_data = data["constraints"]
        return HklCalculation(
            UBCalculation.fromdict(data["ubcalc"]),
            Constraints(constraint_data),
        )
