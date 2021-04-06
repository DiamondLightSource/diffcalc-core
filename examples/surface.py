"""Example of calculations for (2+2) surface diffractometer.

Demonstrate creating UB calculation and setting diffractometer constraints
to calculate diffractometer position for different miller indices with
different constraints, e.g. scattering plane and reference vector orientations.
"""

from math import pi
from pprint import pprint
import numpy as np

from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import TODEG, TORAD
from itertools import product


def position_in_range(pos: Position) -> bool:
    """Check diffractometer position is in acceptale range.

    mu in (0, pi/2)
    nu in (0, pi/2)
    delta in (0, pi/2)
    phi in (-pi/2, pi/2)

    Parameters
    ----------
    pos: Position
        Diffractometer postion to be checked.

    Returns
    -------
    bool
        True, if position is in the acceptable range.
    """
    return all((0 < pos.mu < pi / 2, 0 < pos.nu < pi / 2, 0 < pos.delta < pi / 2, -pi/2 < pos.phi < pi/2))


def demo_hkl_positions():
    """Demonstrate calculations of miller indices and diffractometer positions."""
    cons.betain = 3.0
    #all_pos = hklcalc.get_position(0, 0, 1, wavelength)
    #for pos_001, virtual_angles in all_pos:
    #    if position_in_range(pos_001):
    #        print(f"\n\n[0 0 1] -> {pos_001.asdict}")
    #        print(f"\n{pformat(virtual_angles)}")

    for h, k, l in ((0, 0, 1), (1, 0, 1), (1, 0, 2)):
        all_pos = hklcalc.get_position(h, k, l, wavelength)
        print(f"\n{'hkl':<8s}: [{h:1.0f} {k:1.0f} {l:1.0f}]")
        for pos_001, virtual_angles in all_pos:
            if position_in_range(pos_001):
                for angle, val in pos_001.asdict.items():
                    print(f"{angle:<8s}:{val * TODEG:>8.2f}")
                print("-" * 18)
                for angle, val in virtual_angles.items():
                    print(f"{angle:<8s}:{val * TODEG:>8.2f}")

    pos1 = Position(3.00 * TORAD, 7.90 * TORAD, 14.79 * TORAD, 0.0, 0.0, 8.30 * TORAD)
    hkl1 = hklcalc.get_hkl(pos1, wavelength)
    print("\nPosition -> hkl")
    for angle, val in pos1.asdict.items():
        print(f"{angle:<8s}:{val * TODEG:>8.2f}")
    print("-" * 18)
    print(f"\n{'hkl':<8s}: [{hkl1[0]:1.1f} {hkl1[1]:1.1f} {hkl1[2]:1.1f}]")


def demo_scan_betain(h: float, k: float, l: float) -> None:
    """Scan constrained betain incident angle at [h k l] reflection.

    Parameters
    ----------
    h: float
        Miller index
    k: float
        Miller index
    l: float
        Miller index
    """
    print(f"\n\nScanning betain incident angle at [{h} {k} {l}] reflection.\n")
    print(f"{'betain':<8s}{'mu':>12s}{'delta':>12s}{'gamma':>12s}{'phi':>12s}")
    print("-" * 56)
    for beta_in in np.arange(3.0, 4.1, 0.1):
        cons.betain = beta_in
        for pos, virtual_angles in hklcalc.get_position(h, k, l, wavelength):
            if position_in_range(pos):
                print(
                    f"{beta_in:<8.2f}"
                    f"{pos.mu * TODEG:>12.3f}"
                    f"{pos.delta * TODEG:>12.3f}"
                    f"{pos.nu * TODEG:>12.3f}"
                    f"{pos.phi * TODEG:>12.3f}"
                )
    
def demo_energy_scan(h: float, k: float, l: float) -> None:
    """Scan energy using either fixed incident angle or specular position at [h k l] reflection.

    Parameters
    ----------
    h: float
        Miller index
    k: float
        Miller index
    l: float
        Miller index
    """
    print("\n\nScanning energy in 16-18 keV range at betain = 3.0 and betain=betaout constraints "
          f"at [{h} {k} {l}] reflection.\n")
    for con_name, con_value in (("betain", 3.0), ("bin_eq_bout", True)):
        setattr(cons, con_name, con_value) 
        print(f"\n\n{cons}")
        print(f"{'energy':<8s}{'mu':>12s}{'delta':>12s}{'gamma':>12s}{'phi':>12s}{'betain':>12s}{'betaout':>12s}")
        print("-" * 80)
        for en in np.arange(16, 18.1, .2):
            pos, virtual_angles = next(iter(hklcalc.get_position(h, k, l, 12.39842 / en)))
            print(
                f"{en:<8.2f}"
                f"{pos.mu * TODEG:12.3f}"
                f"{pos.delta * TODEG:12.3f}"
                f"{pos.nu * TODEG:12.3f}"
                f"{pos.phi * TODEG:12.3f}"
                f"{virtual_angles['betain'] * TODEG:12.3f}"
                f"{virtual_angles['betaout'] * TODEG:12.3f}"
            )

def demo_scan_qaz(h, k, l):
    """Scan scattering plane azimuthal orientation.

    Parameters
    ----------
    h: float
        Miller index
    k: float
        Miller index
    l: float
        Miller index
    """
    print(f"\n\nScanning qaz, angle of the scattering plane azimuthal orientation at [{h} {k} {l}] reflection\n")
    print(f"{'qaz':<8s}{'mu':>12s}{'delta':>12s}{'gamma':>12s}{'phi':>12s}{'theta':>12s}")
    print("-" * 68)
    for qaz in np.arange(-1, 1.1, .1):
        cons.qaz = qaz
        pos, virtual_angles = next(iter(hklcalc.get_position(h, k, l, wavelength)))
        print(
            f"{qaz:<8.2f}"
            f"{pos.mu * TODEG:12.3f}"
            f"{pos.delta * TODEG:12.3f}"
            f"{pos.nu * TODEG:12.3f}"
            f"{pos.phi * TODEG:12.3f}"
            f"{virtual_angles['theta'] * TODEG:12.3f}"
        )

def demo_scan_hkl():
    """Scan miller indices."""
    print("\n\nScanning h and l indices in (1, 2) range with k = 0\n")
    print(f"{'hkl':<12s}{'mu':>12s}{'delta':>12s}{'gamma':>12s}{'phi':>12s}{'betain':>12s}{'betaout':>12s}{'theta':>12s}")
    print(
        "-" * 96
    )
    for h,l in product(np.arange(1, 2.1, .1), np.arange(1, 2.1, .1)):
        pos, virtual_angles = next(iter(hklcalc.get_position(h, 0, l, wavelength)))
        print(
            f"[{h:2.1f} 0 {l:2.1f}] "
            f"{pos.mu * TODEG:12.3f}"
            f"{pos.delta * TODEG:12.3f}"
            f"{pos.nu * TODEG:12.3f}"
            f"{pos.phi * TODEG:12.3f}"
            f"{virtual_angles['betain'] * TODEG:12.3f}"
            f"{virtual_angles['betaout'] * TODEG:12.3f}"
            f"{virtual_angles['theta'] * TODEG:12.3f}"
        )
    
if __name__ == "__main__":
    ubcalc = UBCalculation("surface")

    ubcalc.set_lattice("SiO2", 4.913, 5.405)

    ubcalc.n_phi = (0, 0, 1)

    # We define reciprocal lattice directions in laboratory frame
    # taking into account 2 deg crystal mismount around mu axis. 
    start_pos = Position(-2 * TORAD, 0, 0, 0, 0, 0)
    ubcalc.add_orientation((0, 0, 1), (0, 0, 1), start_pos, "norm")
    ubcalc.add_orientation((0, 1, 0), (0, 1, 0), start_pos, "plane")
    ubcalc.calc_ub()

    print(f"UBCalculation object representation.\n")
    print(f"{ubcalc}")
    print(f"\nUB matrix defined as a (3, 3) NumPy array.\n")
    pprint(ubcalc.UB)

    # 2+2 sufrace diffractometer consistes of delta, nu, mu and phi angles.
    # Constraining unused eta and chi angles to 0.
    cons = Constraints({"eta": 0, "chi": 0})
    hklcalc = HklCalculation(ubcalc, cons)

    wavelength = 0.689
    
    demo_hkl_positions()

    demo_scan_betain(1, 0, 1)
    
    demo_energy_scan(0, 0, 1)

    # Remove betain = betaout reference constraint to avoid ambiguity
    # when setting qaz detector constraint. 
    del cons.bin_eq_bout
    
    demo_scan_qaz(0, 0, 1)

    # Remove detector constraint before setting reference constraint
    del cons.qaz

    print("\n\nResetting crystal miscut to 0 (i.e. setting identity U matrix)\n")
    ubcalc.set_miscut(None, 0)
    print(f"{ubcalc}")

    demo_scan_betain(1, 0, 1)

    cons.bin_eq_bout = True
    demo_scan_hkl()
    
    del cons.bin_eq_bout
    demo_scan_qaz(1, 1, 1)
    
    #print(f"{'qaz':<8s}{'mu':>12s}{'delta':>12s}{'gamma':>12s}{'phi':>12s}{'betain':>12s}{'betaout':>12s}")
    #print("-" * 80)
    #for qaz in np.arange(-1, 1.1, .1):
    #    cons.qaz = qaz
    #    for pos, virtual_angles in hklcalc.get_position(0, 0, 1, wavelength):
    #        if position_in_range(pos):
    #            print(
    #                f"{qaz:8.2f}"
    #                f"{pos.mu * TODEG:12.3f}"
    #                f"{pos.delta * TODEG:12.3f}"
    #                f"{pos.nu * TODEG:12.3f}"
    #                f"{pos.phi * TODEG:12.3f}"
    #                f"{virtual_angles['betain'] * TODEG:12.3f}"
    #                f"{virtual_angles['betaout'] * TODEG:12.3f}"
    #            )
    #
    #