"""Module handling constraint information for diffractometer calculations."""
from dataclasses import dataclass
from enum import Enum
from itertools import zip_longest
from typing import Collection, Dict, List, Optional, Tuple, Union

from diffcalc.util import Angle, DiffcalcException
from pint import Quantity
from typing_extensions import Literal

CATEGORY = Enum("CATEGORY", "DETECTOR REFERENCE SAMPLE")
TYPE = Enum("TYPE", "VALUE VOID")


@dataclass(eq=False)
class Constraint:
    """Single constraint object definition."""

    name: str
    category: CATEGORY
    type: TYPE
    value: Optional[Union[Angle, Literal["True"]]] = None

    @property
    def active(self) -> bool:
        """Determine if this constraint has a valid value."""
        return self.value is not None


class Constraints:
    """Collection of angle constraints for diffractometer calculations.

    Three constraints are required for calculations of miller indices and
    the corresponding diffractometer positions. Allowed configurations include
    at most one of the reference and the detector type constraint and up to
    three of the sample type constraints.

    List of the available constraint combinations:

        1 x samp, 1 x ref and 1 x det:  all

        2 x samp and 1 x ref:  chi & phi
                               chi & eta
                               chi & mu
                               mu & eta
                               mu & phi
                               eta & phi

        2 x samp and 1 x det:  chi & phi
                               mu & eta
                               mu & phi
                               mu & chi
                               eta & phi
                               eta & chi
                               bisect & mu
                               bisect & eta
                               bisect & omega

        3 x samp:              eta, chi & phi
                               mu, chi & phi
                               mu, eta & phi
                               mu, eta & chi
    """

    def __init__(
        self,
        constraints: Optional[
            Union[
                Dict[str, Union[Angle, Literal["True"]]],
                Collection[Union[Tuple[str, Angle], str]],
            ]
        ] = None,
    ):
        """Object for setting diffractometer angle constraints."""
        self.delta = Constraint("delta", CATEGORY.DETECTOR, TYPE.VALUE)
        self.nu = Constraint("nu", CATEGORY.DETECTOR, TYPE.VALUE)
        self.qaz = Constraint("qaz", CATEGORY.DETECTOR, TYPE.VALUE)
        self.naz = Constraint("naz", CATEGORY.DETECTOR, TYPE.VALUE)

        self.a_eq_b = Constraint("a_eq_b", CATEGORY.REFERENCE, TYPE.VOID)
        self.alpha = Constraint("alpha", CATEGORY.REFERENCE, TYPE.VALUE)
        self.beta = Constraint("beta", CATEGORY.REFERENCE, TYPE.VALUE)
        self.psi = Constraint("psi", CATEGORY.REFERENCE, TYPE.VALUE)
        self.bin_eq_bout = Constraint("bin_eq_bout", CATEGORY.REFERENCE, TYPE.VOID)
        self.betain = Constraint("betain", CATEGORY.REFERENCE, TYPE.VALUE)
        self.betaout = Constraint("betaout", CATEGORY.REFERENCE, TYPE.VALUE)

        self.mu = Constraint("mu", CATEGORY.SAMPLE, TYPE.VALUE)
        self.eta = Constraint("eta", CATEGORY.SAMPLE, TYPE.VALUE)
        self.chi = Constraint("chi", CATEGORY.SAMPLE, TYPE.VALUE)
        self.phi = Constraint("phi", CATEGORY.SAMPLE, TYPE.VALUE)
        self.bisect = Constraint("bisect", CATEGORY.SAMPLE, TYPE.VOID)
        self.omega = Constraint("omega", CATEGORY.SAMPLE, TYPE.VALUE)

        self.all: Tuple[Constraint, ...] = (
            self.delta,
            self.nu,
            self.qaz,
            self.naz,
            self.a_eq_b,
            self.alpha,
            self.beta,
            self.psi,
            self.bin_eq_bout,
            self.betain,
            self.betaout,
            self.mu,
            self.eta,
            self.chi,
            self.phi,
            self.bisect,
            self.omega,
        )
        if constraints is not None:
            if isinstance(constraints, dict):
                self.asdict = constraints
            elif isinstance(constraints, tuple):
                self.astuple = constraints
            elif isinstance(constraints, (list, set)):
                self.astuple = tuple(constraints)
            else:
                raise DiffcalcException(
                    f"Invalid constraint parameter type: {type(constraints)}"
                )

    @property
    def asdict(self) -> Dict[str, Union[Angle, Literal["True"]]]:
        """Get all constrained angle names and values.

        Returns
        -------
        Dict[str, Union[float, bool]]
            Dictionary with all constrained angle names and values.
        """
        return {con.name: con.value for con in self.all if con.active}

    @asdict.setter
    def asdict(self, constraints: Dict[str, Union[Angle, Literal["True"]]]):
        self.clear()
        for con_name, con_value in constraints.items():
            if hasattr(self, con_name):
                constraint: Constraint = getattr(self, con_name)
                self.set_constraint(constraint, con_value)
            else:
                raise DiffcalcException(f"Invalid constraint name: {con_name}")

    @property
    def astuple(
        self,
    ) -> Tuple[Union[Tuple[str, Union[Angle, Literal["True"]]], str], ...]:
        """Get all constrained angle names and values.

        Returns
        -------
        Tuple[Union[Tuple[str, float], str], ...]
            Tuple with all constrained angle names and values.
        """
        res = []
        for con in self.constrained:
            if con.type is TYPE.VALUE:
                res.append((con.name, getattr(self, con.name).value))
            else:
                res.append(con.name)
        return tuple(res)

    @astuple.setter
    def astuple(
        self,
        constraints: Tuple[Union[Tuple[str, Union[Angle, Literal["True"]]], str], ...],
    ) -> None:
        self.clear()
        for con in constraints:

            con_name = con if isinstance(con, str) else con[0]
            con_value: Union[Angle, Literal["True"]] = (
                True if isinstance(con, str) else con[1]
            )

            if hasattr(self, con_name):
                constraint: Constraint = getattr(self, con_name)
                self.set_constraint(constraint, con_value)
            else:
                raise DiffcalcException(f"Invalid constraint parameter: {con_name}")

    def set_constraint(
        self, con: Constraint, value: Optional[Union[Angle, Literal["True"]]]
    ) -> None:
        """Set a single constraint."""

        def set_value(val: Optional[Union[Angle, Literal["True"]]]) -> None:
            if con.type == TYPE.VALUE:
                if isinstance(val, Quantity):
                    if not val.dimensionless:
                        raise DiffcalcException(
                            f"Non dimensionless units found for {con.name} constraint."
                            f" Please use .deg or .rad units from the diffcalc.ureg "
                            "registry."
                        )
                elif val is True:
                    raise DiffcalcException(
                        f"Constraint {con.name} requires numerical value. "
                        'Found "True" instead.'
                    )

            if con.type == TYPE.VOID:
                if ((val) is not None) and (val is not True):
                    raise DiffcalcException(
                        f"Constraint {con.name} requires boolean value. "
                        f"Found {type(val)} instead."
                    )

            con.value = val

        cons_of_this_category = {
            c for c in self.constrained if c.category is con.category
        }
        can_set_constraint_of_this_category = (
            not self.is_fully_constrained(con) and not self.is_fully_constrained()
        )

        if not can_set_constraint_of_this_category:
            if len(cons_of_this_category) == 1:
                existing_con = cons_of_this_category.pop()
                existing_con.value = None
            else:
                raise DiffcalcException(
                    f"Cannot set {con.name} constraint. First un-constrain one of the\n"
                    f"angles {', '.join(sorted(c.name for c in self.constrained))}."
                )

        set_value(value)

    @property
    def constrained(self):
        """Produce tuple of all active constraints."""
        return tuple(con for con in self.all if con.active)

    @property
    def detector(self) -> Dict[str, Union[Angle, Literal["True"]]]:
        """Produce dictionary of all active detector constraints."""
        return {
            con.name: con.value
            for con in self.all
            if con.active and con.category is CATEGORY.DETECTOR
        }

    @property
    def reference(self) -> Dict[str, Union[Angle, Literal["True"]]]:
        """Produce dictionary of all active reference constraints."""
        return {
            con.name: con.value
            for con in self.all
            if con.active and con.category is CATEGORY.REFERENCE
        }

    @property
    def sample(self) -> Dict[str, Union[Angle, Literal["True"]]]:
        """Produce dictionary of all active sample constraints."""
        return {
            con.name: con.value
            for con in self.all
            if con.active and con.category is CATEGORY.SAMPLE
        }

    def is_fully_constrained(self, con: Optional[Constraint] = None) -> bool:
        """Check if configuration is fully constrained.

        Parameters
        ----------
        con: _Constraint, default = None
            Check if there are available constraints is the same category as the
            input constraint. If parameter is None, check for all constraint
            categories.

        Returns
        -------
        bool:
            True if there aren't any constraints available either in the input
            constraint category or no constraints are available.
        """
        if con is None:
            return len(self.constrained) >= 3

        _max_constrained = {
            CATEGORY.DETECTOR: 1,
            CATEGORY.REFERENCE: 1,
            CATEGORY.SAMPLE: 3,
        }
        count_constrained = len(
            {c for c in self.constrained if c.category is con.category}
        )
        return count_constrained >= _max_constrained[con.category]

    def is_current_mode_implemented(self) -> bool:
        """Check if current constraint set is implemented.

        Configuration needs to be fully constraint for this method to work.

        Returns
        -------
        bool:
            True if current constraint set is supported.
        """
        if not self.is_fully_constrained():
            raise ValueError("Three constraints required")

        count_detector = len(
            {c for c in self.constrained if c.category is CATEGORY.DETECTOR}
        )
        count_reference = len(
            {c for c in self.constrained if c.category is CATEGORY.REFERENCE}
        )
        count_sample = len(
            {c for c in self.constrained if c.category is CATEGORY.SAMPLE}
        )
        if count_sample == 3:
            if (
                set(self.constrained) == {self.chi, self.phi, self.eta}
                or set(self.constrained) == {self.chi, self.phi, self.mu}
                or set(self.constrained) == {self.chi, self.eta, self.mu}
                or set(self.constrained) == {self.phi, self.eta, self.mu}
            ):
                return True
            return False

        if count_sample == 1:
            return self.omega not in set(self.constrained) and self.bisect not in set(
                self.constrained
            )

        if count_reference == 1:
            return (
                {self.chi, self.phi}.issubset(self.constrained)
                or {self.chi, self.eta}.issubset(self.constrained)
                or {self.chi, self.mu}.issubset(self.constrained)
                or {self.mu, self.eta}.issubset(self.constrained)
                or {self.mu, self.phi}.issubset(self.constrained)
                or {self.eta, self.phi}.issubset(self.constrained)
            )

        if count_detector == 1:
            return (
                {self.chi, self.phi}.issubset(self.constrained)
                or {self.mu, self.eta}.issubset(self.constrained)
                or {self.mu, self.phi}.issubset(self.constrained)
                or {self.mu, self.chi}.issubset(self.constrained)
                or {self.eta, self.phi}.issubset(self.constrained)
                or {self.eta, self.chi}.issubset(self.constrained)
                or {self.mu, self.bisect}.issubset(self.constrained)
                or {self.eta, self.bisect}.issubset(self.constrained)
                or {self.omega, self.bisect}.issubset(self.constrained)
            )

        return False

    def clear(self) -> None:
        """Remove all constraints.

        Set all of the constraints as inactive, by setting their value as None.
        """
        for con in self.all:
            con.value = None

        return

    def constrain(
        self, con_name: str, value: Optional[Union[Angle, Literal["True"]]] = None
    ) -> None:
        """Set the value of a single constraint."""
        if hasattr(self, con_name):
            constraint: Constraint = getattr(self, con_name)
            self.set_constraint(constraint, value)
        else:
            raise DiffcalcException(f"Invalid constraint name: {con_name}")

    def unconstrain(self, con_name: str):
        """Unset the value of a single constraint."""
        if hasattr(self, con_name):
            constraint: Constraint = getattr(self, con_name)
            constraint.value = None
        else:
            raise DiffcalcException(f"Invalid constraint name: {con_name}")

    def __str__(self) -> str:
        """Output text representation of active constraint set.

        Returns
        -------
        str
            Table representation of the available constraints with a list of
            constrained angle names and values.
        """
        lines = []
        lines.extend(self._build_display_table_lines())
        lines.append("")
        lines.extend(self._report_constraints_lines())
        lines.append("")
        if self.is_fully_constrained() and not self.is_current_mode_implemented():
            lines.append("    Sorry, this constraint combination is not implemented.")
        return "\n".join(lines)

    def _build_display_table_lines(self) -> List[str]:
        constraint_types = [
            (self.delta, self.nu, self.qaz, self.naz),
            (
                self.a_eq_b,
                self.alpha,
                self.beta,
                self.psi,
                self.bin_eq_bout,
                self.betain,
                self.betaout,
            ),
            (self.mu, self.eta, self.chi, self.phi, self.bisect, self.omega),
        ]
        max_name_width = max(len(con.name) for con in self.all)

        cells = []

        header_cells = []
        header_cells.append("    " + "DET".ljust(max_name_width))
        header_cells.append("    " + "REF".ljust(max_name_width))
        header_cells.append("    " + "SAMP")
        cells.append(header_cells)

        underline_cells = ["    " + "-" * max_name_width] * len(constraint_types)
        cells.append(underline_cells)

        for con_line in zip_longest(*constraint_types):
            row_cells = []
            for con in con_line:
                name = con.name if con is not None else ""
                row_cells.append("   " if con is None or not con.active else "-->")
                row_cells.append(("%-" + str(max_name_width) + "s") % name)
            cells.append(row_cells)

        lines = [" ".join(row_cells).rstrip() for row_cells in cells]
        return lines

    def _report_constraint(self, con: Constraint) -> str:
        val = con.value
        if con.type is TYPE.VOID:
            return "    %s" % con.name

        if isinstance(val, Quantity):
            return f"    {con.name:<5}: {val.magnitude:.4f}"
        else:
            return f"    {con.name:<5}: {val:.4f}"

    def _report_constraints_lines(self) -> List[str]:
        lines = []
        required = 3 - len(self.constrained)
        if required == 0:
            pass
        elif required == 1:
            lines.append("!   1 more constraint required")
        else:
            lines.append("!   %d more constraints required" % required)
        lines.extend([self._report_constraint(con) for con in self.all if con.active])
        return lines
