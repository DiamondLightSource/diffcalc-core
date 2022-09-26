"""Module handling constraint information for diffractometer calculations."""
import dataclasses
from enum import Enum
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple, Union

from diffcalc.util import DiffcalcException
from typing_extensions import Literal


class ConstraintTypes(Enum):
    SAMPLE = 1
    DETECTOR = 5
    REFERENCE = 11


detector_constraints = Enum("DETCONS", "DELTA NU QAZ NAZ")
reference_constraints = Enum(
    "REFCONS", "A_EQ_B ALPHA BETA PSI BIN_EQ_BOUT BETAIN BETAOUT"
)
sample_constraints = Enum("SAMPCONS", "MU ETA CHI PHI BISECT OMEGA")

boolean_constraints = Enum("BOOLCONS", "A_EQ_B BIN_EQ_BOUT BISECT")


@dataclasses.dataclass
class Constraints:
    cons_as_dict: dataclasses.InitVar[Optional[Dict[str, Union[float, bool]]]] = None

    delta: Optional[float] = None
    nu: Optional[float] = None
    qaz: Optional[float] = None
    naz: Optional[float] = None
    a_eq_b: Optional[bool] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    psi: Optional[float] = None
    bin_eq_bout: Optional[bool] = None
    betain: Optional[float] = None
    betaout: Optional[float] = None
    mu: Optional[float] = None
    eta: Optional[float] = None
    chi: Optional[float] = None
    phi: Optional[float] = None
    bisect: Optional[bool] = None
    omega: Optional[float] = None

    def __post_init__(self, cons_as_dict) -> None:
        if not cons_as_dict:
            return

        for constraint_name, constraint_value in cons_as_dict.items():
            self._set_attr(constraint_name, constraint_value)

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
        active = self.asdict
        constraint_names = [
            tuple([i.name.lower() for i in detector_constraints]),
            tuple([i.name.lower() for i in reference_constraints]),
            tuple([i.name.lower() for i in sample_constraints]),
        ]
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

        constraints_names_and_values = [
            list(zip(constraint_names[i], constraint_types[i]))
            for i in range(len(ConstraintTypes))
        ]
        max_name_width = max([len(con) for con in vars(self).keys()])

        cells = []

        header_cells = []
        header_cells.append("    " + "DET".ljust(max_name_width))
        header_cells.append("    " + "REF".ljust(max_name_width))
        header_cells.append("    " + "SAMP")
        cells.append(header_cells)

        underline_cells = ["    " + "-" * max_name_width] * len(constraint_types)
        cells.append(underline_cells)

        for con_line in zip_longest(*constraints_names_and_values):
            row_cells = []
            for con in con_line:
                name = con[0] if con is not None else ""
                row_cells.append(
                    "   " if con is None or not (con[0] in active) else "-->"
                )
                row_cells.append(("%-" + str(max_name_width) + "s") % name)
            cells.append(row_cells)

        lines = [" ".join(row_cells).rstrip() for row_cells in cells]
        return lines

    def _report_constraint(self, con: str) -> str:
        val = getattr(self, con)
        if con in [c.name.lower() for c in boolean_constraints]:
            return "    %s" % con
        else:
            return f"    {con:<5}: {val:.4f}"

    def _report_constraints_lines(self) -> List[str]:
        lines = []
        required = 3 - len(self.asdict)
        if required == 0:
            pass
        elif required == 1:
            lines.append("!   1 more constraint required")
        else:
            lines.append("!   %d more constraints required" % required)
        lines.extend([self._report_constraint(con) for con in self.asdict.keys()])
        return lines

    def _set_attr(self, attr: str, value: Optional[Union[float, bool]]):
        active_cons = self.asdict

        names, categories = self._constraint_types(attr)
        attr_set_to_None = self._set_correct_value_for_attr(attr, value)

        if (attr_set_to_None) or (attr in names[:-1]):
            return

        ref = [
            idx for idx, c in enumerate(categories) if c == ConstraintTypes.REFERENCE
        ]
        det = [idx for idx, c in enumerate(categories) if c == ConstraintTypes.DETECTOR]
        samp = [idx for idx, c in enumerate(categories) if c == ConstraintTypes.SAMPLE]

        if len(active_cons) == 3:
            if (len(samp) > 2) or (
                (len(samp) == 2) and (categories[-1] not in categories[:-1])
            ):
                raise DiffcalcException(
                    f"Cannot set {attr} constraint. Please unconstrain one of the "
                    + f"angles {names[:-1]}."
                )
            if categories[-1] == ConstraintTypes.SAMPLE:
                self._set_correct_value_for_attr(names[samp[0]], None)

        if len(ref) == 2:
            self._set_correct_value_for_attr(names[ref[0]], None)

        if len(det) == 2:
            self._set_correct_value_for_attr(names[det[0]], None)

    def _set_correct_value_for_attr(
        self, attr: str, value: Optional[Union[bool, float]]
    ) -> int:
        """Returns 1 if value set to None, i.e unconstrained."""

        boolean_cons = [con.name.lower() for con in boolean_constraints]

        if (value is not None) and (attr in boolean_cons):
            if bool(value):
                setattr(self, attr, bool(value))
                return 0
        elif (value is not None) and (attr not in boolean_cons):
            setattr(self, attr, float(value))
            return 0

        setattr(self, attr, None)
        return 1

    def _constraint_types(
        self, attr: str
    ) -> Tuple[
        List[str],
        List[
            Literal[
                ConstraintTypes.DETECTOR,
                ConstraintTypes.REFERENCE,
                ConstraintTypes.SAMPLE,
            ]
        ],
    ]:
        active_cons = self.asdict

        det_names = [con.name.lower() for con in detector_constraints]
        samp_names = [con.name.lower() for con in sample_constraints]
        ref_names = [con.name.lower() for con in reference_constraints]

        con_type = []
        con_name = []

        for constraint, _ in active_cons.items():
            if constraint in det_names:
                con_type.append(ConstraintTypes.DETECTOR)
            elif constraint in samp_names:
                con_type.append(ConstraintTypes.SAMPLE)
            elif constraint in ref_names:
                con_type.append(ConstraintTypes.REFERENCE)
            con_name.append(constraint)

        if attr in det_names:
            con_type.append(ConstraintTypes.DETECTOR)
        elif attr in samp_names:
            con_type.append(ConstraintTypes.SAMPLE)
        elif attr in ref_names:
            con_type.append(ConstraintTypes.REFERENCE)
        else:
            raise DiffcalcException(f"invalid constraint given: {attr}")
        con_name.append(attr)

        return con_name, con_type

    @property
    def detector(self) -> Dict[str, float]:
        det_names = [con.name.lower() for con in detector_constraints]
        active_cons = self.asdict

        return {
            i: active_cons.get(i) for i in det_names if active_cons.get(i) is not None
        }

    @property
    def sample(self) -> Dict[str, float]:
        samp_names = [con.name.lower() for con in sample_constraints]
        active_cons = self.asdict

        return {
            i: active_cons.get(i) for i in samp_names if active_cons.get(i) is not None
        }

    @property
    def reference(self) -> Dict[str, float]:
        ref_names = [con.name.lower() for con in reference_constraints]
        active_cons = self.asdict

        return {
            i: active_cons.get(i) for i in ref_names if active_cons.get(i) is not None
        }

    @property
    def asdict(self):
        return {k: v for k, v in vars(self).items() if v is not None}

    def is_fully_constrained(self):
        """Check if the instance is fully constrained."""
        return len(self.asdict) == 3

    def is_current_mode_implemented(self):
        """Check if constraints are valid."""
        det = self.detector
        samp = self.sample
        ref = self.reference

        total_cons = len(ref) + len(samp) + len(det)

        if (len(ref) > 1) or (len(det) > 1):
            return False
        if total_cons < 3:
            return False

        if len(det) == len(samp) == len(ref):
            return True
        elif len(samp) == 2 and len(det) == 1:
            if "bisect" in samp:
                return ("mu" in samp) or ("eta" in samp) or ("omega" in samp)

        return all([con in {"mu", "eta", "chi", "phi"} for con in samp.keys()])

    def constrain(self, attr: str, value: Optional[Union[float, bool]]):
        """Set or unset a constraint value."""
        self._set_attr(attr, value)

    def unconstrain(self, attr: str):
        self.constrain(attr, None)

    def clear(self):
        active_cons = self.asdict

        for con_name, _ in active_cons.items():
            self.unconstrain(con_name)
