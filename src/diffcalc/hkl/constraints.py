"""Module handling constraint information for diffractometer calculations."""
import dataclasses
from enum import Enum
from itertools import zip_longest
from typing import Dict, List, Literal, Optional, Tuple, Union

from diffcalc.util import DiffcalcException


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

    def __post_init__(self, cons_as_dict) -> None:
        self.delta: Optional[float] = None
        self.nu: Optional[float] = None
        self.qaz: Optional[float] = None
        self.naz: Optional[float] = None
        self.a_eq_b: Optional[bool] = None
        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None
        self.psi: Optional[float] = None
        self.bin_eq_bout: Optional[bool] = None
        self.betain: Optional[float] = None
        self.betaout: Optional[float] = None
        self.mu: Optional[float] = None
        self.eta: Optional[float] = None
        self.chi: Optional[float] = None
        self.phi: Optional[float] = None
        self.bisect: Optional[bool] = None
        self.omega: Optional[float] = None

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


"""
# what happens if you give:
# 1. one of each? (3)
fake = FakeConstraints({"delta": 1.0, "a_eq_b": True, "mu": 1.0})
fake = FakeConstraints({"delta": 1.0, "alpha": 1.0, "mu": 1.0})

# a) a sample on top
fake = FakeConstraints({"delta": 1.0, "a_eq_b": True, "mu": 1.0, "phi": 1.0})
fake = FakeConstraints({"delta": 1.0, "alpha": 1.0, "mu": 1.0, "phi": 1.0})

# b) a detector on top
fake = FakeConstraints({"delta": 1.0, "a_eq_b": True, "mu": 1.0, "nu": 1.0})

fake = FakeConstraints({"delta": 1.0, "alpha": 1.0, "mu": 1.0, "nu": 1.0})
real = Constraints({"delta": 1.0, "alpha": 1.0, "mu": 1.0, "nu": 1.0})

# c) a ref on top
fake = FakeConstraints({"delta": 1.0, "a_eq_b": True, "mu": 1.0, "beta": 1.0})
real = Constraints({"delta": 1.0, "a_eq_b": True, "mu": 1.0, "beta": 1.0})

fake = FakeConstraints({"delta": 1.0, "alpha": 1.0, "mu": 1.0, "beta": 1.0})
real = Constraints({"delta": 1.0, "alpha": 1.0, "mu": 1.0, "beta": 1.0})
############################################################################

# 2. three dets
fake = FakeConstraints({"delta": 1.0, "nu": 1.0, "qaz": 1.0})
real = Constraints({"delta": 1.0, "nu": 1.0, "qaz": 1.0})

# a) sample on top
fake = FakeConstraints({"delta": 1.0, "nu": 1.0, "qaz": 1.0, "phi": 1.0})
real = Constraints({"delta": 1.0, "nu": 1.0, "qaz": 1.0, "phi": 1.0})

# b) detector on top
fake = FakeConstraints({"delta": 1.0, "nu": 1.0, "qaz": 1.0, "naz": 1.0})
real = Constraints({"delta": 1.0, "nu": 1.0, "qaz": 1.0, "naz": 1.0})

# c) ref on top
fake = FakeConstraints({"delta": 1.0, "nu": 1.0, "qaz": 1.0, "beta": 1.0})
real = Constraints({"delta": 1.0, "nu": 1.0, "qaz": 1.0, "beta": 1.0})
############################################################################

# 3. three refs
fake = FakeConstraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0})
real = Constraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0})

# a) sample
fake = FakeConstraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0, "phi": 1.0})
real = Constraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0, "phi": 1.0})

fake = FakeConstraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0, "bisect": True})
real = Constraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0, "bisect": True})

# b) detector
fake = FakeConstraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0, "nu": 1.0})
real = Constraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0, "nu": 1.0})

# c) ref
fake = FakeConstraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0, "bin_eq_bout": True})
real = Constraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0, "bin_eq_bout": True})
fake = FakeConstraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0, "betain": 1.0})
real = Constraints({"alpha": 1.0, "beta": 1.0, "psi": 1.0, "betain": 1.0})

############################################################################
# 4. three samples
fake = FakeConstraints({"mu": 1.0, "eta": 1.0, "chi": 1.0})
real = Constraints({"mu": 1.0, "eta": 1.0, "chi": 1.0})

# a) sample
fake = FakeConstraints(
    {"mu": 1.0, "eta": 1.0, "chi": 1.0, "phi": 1.0}
)
real = Constraints({"mu": 1.0, "eta": 1.0, "chi": 1.0, "phi": 1.0})

# b) detector
fake = FakeConstraints(
    {"mu": 1.0, "eta": 1.0, "chi": 1.0, "qaz": 1.0}
)  # this is wrong!
real = Constraints({"mu": 1.0, "eta": 1.0, "chi": 1.0, "qaz": 1.0})

# c) ref
fake = FakeConstraints({"mu": 1.0, "eta": 1.0, "chi": 1.0, "psi": 1.0})
real = Constraints({"mu": 1.0, "eta": 1.0, "chi": 1.0, "psi": 1.0})

############################################################################
# 5. 2 samples and a det
fake = FakeConstraints({"mu": 1.0, "eta": 1.0, "delta": 1.0})
real = Constraints({"mu": 1.0, "eta": 1.0, "delta": 1.0})

# a) sample
fake = FakeConstraints({"mu": 1.0, "eta": 1.0, "delta": 1.0, "chi": 1.0})
real = Constraints({"mu": 1.0, "eta": 1.0, "delta": 1.0, "chi": 1.0})

# b) detector
fake = FakeConstraints({"mu": 1.0, "eta": 1.0, "delta": 1.0, "nu": 1.0})
real = Constraints({"mu": 1.0, "eta": 1.0, "delta": 1.0, "nu": 1.0})

# c) ref
fake = FakeConstraints({"mu": 1.0, "eta": 1.0, "delta": 1.0, "alpha": 1.0})
real = Constraints({"mu": 1.0, "eta": 1.0, "delta": 1.0, "alpha": 1.0})

############################################################################
# 6. 2 samples and a ref
fake = FakeConstraints({"mu": 1.0, "eta": 1.0, "alpha": 1.0})
real = Constraints({"mu": 1.0, "eta": 1.0, "alpha": 1.0})

# a) sample
fake = FakeConstraints({"mu": 1.0, "eta": 1.0, "alpha": 1.0, "chi": 1.0})
real = Constraints({"mu": 1.0, "eta": 1.0, "alpha": 1.0, "chi": 1.0})

# b) detector
fake = FakeConstraints({"mu": 1.0, "eta": 1.0, "alpha": 1.0, "delta": 1.0})
real = Constraints({"mu": 1.0, "eta": 1.0, "alpha": 1.0, "delta": 1.0})

# c) ref
fake = FakeConstraints({"mu": 1.0, "eta": 1.0, "alpha": 1.0, "beta": 1.0})
real = Constraints({"mu": 1.0, "eta": 1.0, "alpha": 1.0, "beta": 1.0})
# 7. one on top each of those options? (specifically 5 +6 )

# just here for reference
# detector_constraints = Enum("DETCONS", "DELTA NU QAZ NAZ")
# reference_constraints = Enum(
#     "REFCONS", "A_EQ_B ALPHA BETA PSI BIN_EQ_BOUT BETAIN BETAOUT"
# )
# sample_constraints = Enum("SAMPCONS", "MU ETA CHI PHI BISECT OMEGA")

# boolean_constraints = Enum("BOOLCONS", "A_EQ_B BIN_EQ_BOUT BISECT")


FakeConstraints({"delta": 1.0, "nu": 1.0, "qaz": 1.0})

wat = FakeConstraints({"alpha": 1.0, "beta": 2.0})
print("wat")
"""
