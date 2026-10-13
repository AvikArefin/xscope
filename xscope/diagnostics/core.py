import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

from xscope.data_manager import RunDataManager
from xscope.diagnostics.series import (
    MetricInfo, Trend, X_AXIS_KEYS, build_trend, classify_metric, is_finite_number,
)

logger = logging.getLogger("xscope.diagnostics")

SEVERITY_ORDER = {'critical': 0, 'warning': 1, 'info': 2}


@dataclass
class DiagnosticResult:
    severity: str  # 'critical', 'warning', 'info'
    title: str
    message: str
    run_name: str
    epoch: Optional[int] = None
    suggested_action: Optional[str] = None
    rule: str = ""
    metric: Optional[str] = None

    def key(self) -> tuple:
        return (self.run_name, self.title, self.metric, self.epoch)


@dataclass
class DiagnosticState:
    seen: set = field(default_factory=set)
    dismissed: set = field(default_factory=set)


def run_label(run: dict) -> str:
    """'v3/100 #5': group (if the run has one), name and experiment number."""
    exp_name = str(run.get('experiment_name', ''))
    exp_num = run.get('experiment_number', '')
    if run.get('group'):
        exp_name = f"{run['group']}/{exp_name}"
    if exp_num != "":
        return f"{exp_name} #{exp_num}"
    return exp_name or run.get('run_path', 'unknown')


class RunContext:
    """Per-run view handed to every rule: cached records, metric roles and trends.

    Rules may also call each other's detectors through `memo()` so that, e.g., the
    plateau rule can stay quiet for a run that the overfitting rule already explains.
    """

    def __init__(self, run: dict, data_manager: RunDataManager):
        self.run = run
        self.data_manager = data_manager
        self.name = run_label(run)
        self.results: list[DiagnosticResult] = []  # filled by per-run rules, read by cross-run rules
        self._memo: dict = {}

    @property
    def run_path(self) -> str:
        return self.run.get('run_path', '')

    def records(self, filename: str = "metrics.jsonl") -> list[dict]:
        return self.memo(('records', filename), lambda: self.data_manager.get_records(self.run_path, filename)[1])

    def json_file(self, filename: str) -> Optional[dict]:
        """Reads a small JSON file from the run directory (config.json, status.json); None if absent/invalid."""
        def compute():
            path = os.path.join(self.run_path, filename)
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else None
            except (OSError, ValueError):
                return None
        return self.memo(('json', filename), compute)

    def config(self) -> dict:
        return self.json_file("config.json") or {}

    def status(self) -> Optional[dict]:
        return self.json_file("status.json")

    def times(self, filename: str = "metrics.jsonl") -> list[float]:
        """Wall-clock '_time' of each record (written by current mtrick; absent in older runs)."""
        return [r['_time'] for r in self.records(filename) if is_finite_number(r.get('_time'))]

    def memo(self, key, compute: Callable):
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    def metrics(self) -> dict[str, MetricInfo]:
        """Numeric metric keys in metrics.jsonl, classified by role."""
        def compute():
            keys: dict[str, MetricInfo] = {}
            for r in self.records():
                for k, v in r.items():
                    if k in keys or k.lower() in X_AXIS_KEYS or k.startswith('_'):
                        continue
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        keys[k] = classify_metric(k)
            return keys
        return self.memo('metrics', compute)

    def series(self, key: str) -> tuple[list[float], list[float]]:
        """(xs, ys) for a metric, skipping records where it is missing or non-finite."""
        def compute():
            xs, ys = [], []
            for i, r in enumerate(self.records()):
                v = r.get(key)
                if is_finite_number(v):
                    x = r.get('epoch', r.get('step', i + 1))
                    xs.append(x if is_finite_number(x) else i + 1)
                    ys.append(float(v))
            return xs, ys
        return self.memo(('series', key), compute)

    def trend(self, key: str) -> Optional[Trend]:
        def compute():
            info = self.metrics().get(key)
            xs, ys = self.series(key)
            if info is None or len(ys) < 3:
                return None
            return build_trend(key, xs, ys, info.lower_is_better)
        return self.memo(('trend', key), compute)

    def metrics_of_kind(self, kind: str) -> list[MetricInfo]:
        return [m for m in self.metrics().values() if m.kind == kind]

    def pairs(self, kind: str) -> list[tuple[MetricInfo, MetricInfo]]:
        """(train, val) metric pairs of the given kind sharing the same base name."""
        by_base: dict[str, dict[str, MetricInfo]] = {}
        for m in self.metrics_of_kind(kind):
            if m.split:
                by_base.setdefault(m.base, {}).setdefault(m.split, m)
        return [(d['train'], d['val']) for d in by_base.values() if 'train' in d and 'val' in d]

    def primary_losses(self) -> list[MetricInfo]:
        """Losses that describe optimisation: train or unsplit losses, plus val losses with no train twin."""
        losses = self.metrics_of_kind('loss')
        train_bases = {m.base for m in losses if m.split == 'train'}
        return [m for m in losses if m.split != 'val' or m.base not in train_bases]


RuleFn = Callable[[RunContext], object]
CrossRuleFn = Callable[[list[RunContext]], object]

DIAGNOSTIC_RULES: dict[str, RuleFn] = {}
CROSS_RUN_RULES: dict[str, CrossRuleFn] = {}


def _register(registry: dict, name: str, description: str, enabled: bool):
    def decorator(func):
        rule_name = name or func.__name__
        func._diag_meta = {"name": rule_name, "description": description, "enabled": enabled}
        registry[rule_name] = func
        return func
    return decorator


def diagnostic_rule(name: str = "", description: str = "", enabled: bool = True):
    """Registers a per-run rule: `fn(ctx: RunContext) -> DiagnosticResult | list | None`."""
    return _register(DIAGNOSTIC_RULES, name, description, enabled)


def cross_run_diagnostic_rule(name: str = "", description: str = "", enabled: bool = True):
    """Registers a cross-run rule: `fn(contexts: list[RunContext]) -> DiagnosticResult | list | None`."""
    return _register(CROSS_RUN_RULES, name, description, enabled)


def _as_list(out) -> Iterable[DiagnosticResult]:
    if out is None:
        return []
    if isinstance(out, DiagnosticResult):
        return [out]
    return [r for r in out if r is not None]


def run_diagnostics_for_runs(
    selected_runs: list[dict],
    data_manager: RunDataManager,
    state: Optional[DiagnosticState] = None,
) -> List[DiagnosticResult]:
    """Evaluates all registered diagnostic rules across the specified selected runs."""
    results: List[DiagnosticResult] = []
    contexts = [RunContext(run, data_manager) for run in selected_runs]

    for ctx in contexts:
        for rule_name, rule in DIAGNOSTIC_RULES.items():
            if not rule._diag_meta.get("enabled", True):
                continue
            try:
                for res in _as_list(rule(ctx)):
                    res.rule = res.rule or rule_name
                    ctx.results.append(res)
                    results.append(res)
            except Exception:
                logger.exception("Error in diagnostic rule '%s' for run '%s'", rule_name, ctx.name)

    # Cross-run rules see every selected run; each decides how many runs it needs.
    for rule_name, rule in CROSS_RUN_RULES.items():
        if not rule._diag_meta.get("enabled", True):
            continue
        try:
            for res in _as_list(rule(contexts)):
                res.rule = res.rule or rule_name
                results.append(res)
        except Exception:
            logger.exception("Error in cross-run diagnostic rule '%s'", rule_name)

    if state:
        results = [r for r in results if r.key() not in state.dismissed]
        state.seen.update(r.key() for r in results)

    results.sort(key=lambda r: (SEVERITY_ORDER.get(r.severity, 3), r.run_name))
    return results
