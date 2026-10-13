"""Automated diagnostics for runs logged with mtrick.

Rules register themselves on import; `run_diagnostics_for_runs` evaluates them all.
"""
from xscope.diagnostics.core import (
    CROSS_RUN_RULES,
    DIAGNOSTIC_RULES,
    DiagnosticResult,
    DiagnosticState,
    RunContext,
    cross_run_diagnostic_rule,
    diagnostic_rule,
    run_diagnostics_for_runs,
)
from xscope.diagnostics import (  # noqa: F401  (importing registers the rules)
    rules_scalar, rules_lifecycle, rules_matrix, rules_2d, rules_calibration, rules_cross,
)

__all__ = [
    "CROSS_RUN_RULES",
    "DIAGNOSTIC_RULES",
    "DiagnosticResult",
    "DiagnosticState",
    "RunContext",
    "cross_run_diagnostic_rule",
    "diagnostic_rule",
    "run_diagnostics_for_runs",
]
