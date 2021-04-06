"""Example of calculations for (4+2) diffractometer.

Demonstrate creating UB calculation and setting diffractometer constraints
to calculate diffractometer position for different miller indices with
different constraints, e.g. scattering plane and reference vector orientations.
"""

from itertools import product
from math import pi
from pprint import pprint

import numpy as np
from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import TODEG, TORAD


def in_range_mu_nu_phi(pos: Position) -> bool:
    """Check diffractometer position is in acceptale np.arange.

    mu in (0, pi/2)
    nu in (0, pi/2)
    phi in (-pi/2, pi/2)

    Parameters
    ----------
    pos: Position
        Diffractometer postion to be checked.

    Returns
    -------
    bool
        True, if position is in the acceptable np.arange.
    """
    return all((0 < pos.mu < pi / 2, 0 < pos.nu < pi / 2, -pi / 2 < pos.phi < pi / 2))


def in_range_del_eta_phi(pos: Position) -> bool:
    """Check diffractometer position is in acceptale np.arange.

    delta in (-pi/2, pi/2)
    eta in (-pi/2, pi/2)
    phi in (-pi/2, pi/2)

    Parameters
    ----------
    pos: Position
        Diffractometer postion to be checked.

    Returns
    -------
    bool
        True, if position is in the acceptable np.arange.
    """
    return all(
        (
            -pi / 2 < pos.delta < pi / 2,
            -pi / 2 < pos.eta < pi / 2,
            -pi / 2 < pos.phi < pi / 2,
        )
    )


def get_hkl_positions():
    """Demonstrate calculations of miller indices and diffractometer positions."""
    for h, k, l in ((0, 0, 1), (0, 1, 1), (1, 0, 2)):
        all_pos = hklcalc.get_position(h, k, l, wavelength)
        print(f"\n{'hkl':<8s}: [{h:1.0f} {k:1.0f} {l:1.0f}]")
        for pos_001, virtual_angles in all_pos:
            if in_range_mu_nu_phi(pos_001):
                for angle, val in pos_001.asdict.items():
                    print(f"{angle:<8s}:{val * TODEG:>8.2f}")
                print("-" * 18)
                for angle, val in virtual_angles.items():
                    print(f"{angle:<8s}:{val * TODEG:>8.2f}")

    pos1 = Position(7.31 * TORAD, 0.0, 10.62 * TORAD, 0, 0.0, 0)
    hkl1 = hklcalc.get_hkl(pos1, wavelength)
    print("\nPosition -> hkl")
    for angle, val in pos1.asdict.items():
        print(f"{angle:<8s}:{val * TODEG:>8.2f}")
    print("-" * 18)
    print(f"\n{'hkl':<8s}: [{hkl1[0]:1.1f} {hkl1[1]:1.1f} {hkl1[2]:1.1f}]")


def demo_scan_hkl():
    """Scan miller indices."""
    print("Scanning h and l indices in (1, 2) range with k = 0\n")
    print(
        f"{'hkl':<12s}{'mu':>12s}{'delta':>12s}{'gamma':>12s}{'eta':>12s}{'chi':>12s}{'phi':>12s}{'alpha':>12s}{'beta':>12s}{'theta':>12s}"
    )
    print("-" * 120)
    for h, l in product(np.arange(1, 2.1, 0.1), np.arange(1, 2.1, 0.1)):
        pos, virtual_angles = next(iter(hklcalc.get_position(h, 0, l, wavelength)))
        print(
            f"[{h:2.1f} 0 {l:2.1f}] "
            f"{pos.mu * TODEG:12.3f}"
            f"{pos.delta * TODEG:12.3f}"
            f"{pos.nu * TODEG:12.3f}"
            f"{pos.eta * TODEG:12.3f}"
            f"{pos.chi * TODEG:12.3f}"
            f"{pos.phi * TODEG:12.3f}"
            f"{virtual_angles['alpha'] * TODEG:12.3f}"
            f"{virtual_angles['beta'] * TODEG:12.3f}"
            f"{virtual_angles['theta'] * TODEG:12.3f}"
        )


def demo_scan_alpha():
    """Scan constrained alpha reference vecor angle."""
    print("\n\nScanning alpha incident angle w.r.t the reference vector.\n")
    print(
        f"{'alpha':<6s}{'mu':>12s}{'gamma':>12s}{'chi':>12s}{'phi':>12s}{'theta':>12s}"
    )
    print("-------------------------------------------------------------")
    for alp in np.arange(0, 11, 1):
        cons.alpha = alp
        for pos, virtual_angles in hklcalc.get_position(0, 0, 1, wavelength):
            if in_range_mu_nu_phi(pos):
                print(
                    f"{virtual_angles['alpha'] * TODEG:6.2f}"
                    f"{pos.mu * TODEG:12.3f}"
                    f"{pos.nu * TODEG:12.3f}"
                    f"{pos.chi * TODEG:12.3f}"
                    f"{pos.phi * TODEG:12.3f}"
                    f"{virtual_angles['theta'] * TODEG:12.3f}"
                )


def demo_scan_qaz():
    """Scan scattering plane azimuthal orientation."""
    print("\n\nScanning qaz, angle of the scattering plane azimuthal orientation\n")
    print(
        f"{'qaz':<6s}{'mu':>12s}{'delta':>12s}{'gamma':>12s}{'chi':>12s}{'phi':>12s}{'psi':>12s}"
    )
    print("-" * 78)
    for qaz in np.arange(90, -1, -10):
        cons.qaz = qaz
        for pos, virtual_angles in hklcalc.get_position(0, 0, 1, wavelength):
            if in_range_mu_nu_phi(pos):
                print(
                    f"{virtual_angles['qaz'] * TODEG:6.2f}"
                    f"{pos.mu * TODEG:12.3f}"
                    f"{pos.delta * TODEG:12.3f}"
                    f"{pos.nu * TODEG:12.3f}"
                    f"{pos.chi * TODEG:12.3f}"
                    f"{pos.phi * TODEG:12.3f}"
                    f"{virtual_angles['psi'] * TODEG:12.3f}"
                )


def demo_scan_psi():
    """Scan reference vector azimuthal orientation with and without crystal mismount."""
    print("\n\nScanning psi, reference vector azimuthal orientation angle\n")
    print(
        f"{'psi':<6s}{'mu':>12s}{'delta':>12s}{'gamma':>12s}{'eta':>12s}{'chi':>12s}{'phi':>12s}{'qaz':>12s}"
    )
    print("-" * 92)
    for psi in np.arange(90, -1, -10):
        cons.psi = psi
        for pos, virtual_angles in hklcalc.get_position(0, 0, 1, wavelength):
            if in_range_del_eta_phi(pos):
                print(
                    f"{virtual_angles['psi'] * TODEG:6.2f}"
                    f"{pos.mu * TODEG:12.3f}"
                    f"{pos.delta * TODEG:12.3f}"
                    f"{pos.nu * TODEG:12.3f}"
                    f"{pos.eta * TODEG:12.3f}"
                    f"{pos.chi * TODEG:12.3f}"
                    f"{pos.phi * TODEG:12.3f}"
                    f"{virtual_angles['qaz'] * TODEG:12.3f}"
                )

    print("\n\nResetting crystal miscut to 0 (i.e. setting identity U matrix)\n")
    ubcalc.set_miscut(None, 0)
    print(f"\n{ubcalc}\n\n")
    print(
        f"{'psi':<6s}{'mu':>12s}{'delta':>12s}{'gamma':>12s}{'eta':>12s}{'chi':>12s}{'phi':>12s}{'qaz':>12s}"
    )
    print("-" * 92)
    for psi in np.arange(90, -1, -10):
        cons.psi = psi
        for pos, virtual_angles in hklcalc.get_position(0, 0, 1, wavelength):
            if in_range_del_eta_phi(pos):
                print(
                    f"{virtual_angles['psi'] * TODEG:6.2f}"
                    f"{pos.mu * TODEG:12.3f}"
                    f"{pos.delta * TODEG:12.3f}"
                    f"{pos.nu * TODEG:12.3f}"
                    f"{pos.eta * TODEG:12.3f}"
                    f"{pos.chi * TODEG:12.3f}"
                    f"{pos.phi * TODEG:12.3f}"
                    f"{virtual_angles['qaz'] * TODEG:12.3f}"
                )


if __name__ == "__main__":
    ubcalc = UBCalculation("sixcircle")

    ubcalc.set_lattice("SiO2", 4.913, 5.405)

    ubcalc.n_hkl = (1, 0, 0)

    ubcalc.add_reflection(
        (0, 0, 1),
        Position(7.31 * TORAD, 0.0, 10.62 * TORAD, 0, 0.0, 0),
        12.39842,
        "refl1",
    )
    ubcalc.add_orientation((0, 1, 0), (0, 1, 0), None, "plane")
    ubcalc.calc_ub("refl1", "plane")

    print(f"UBCalculation object representation.\n")
    print(f"{ubcalc}")
    print(f"\nUB matrix defined as a (3, 3) NumPy array.\n")
    pprint(ubcalc.UB)

    cons = Constraints({"qaz": 0, "alpha": 0, "eta": 0})
    hklcalc = HklCalculation(ubcalc, cons)

    wavelength = 1.0

    get_hkl_positions()

    print(
        "\n\nConstraining incident and exit angles w.r.t. reference vector to be equal (a_eq_b)."
    )
    cons.a_eq_b = True
    demo_scan_hkl()

    demo_scan_alpha()

    demo_scan_qaz()

    cons.asdict = {"qaz": 90, "chi": 90}
    demo_scan_psi()
