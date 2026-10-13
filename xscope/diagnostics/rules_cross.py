"""Diagnostics that compare the selected runs with each other."""
import math
import statistics

from xscope.diagnostics.core import DiagnosticResult, RunContext, cross_run_diagnostic_rule
from xscope.diagnostics.rules_matrix import MATRIX_FILE, parse_confusion
from xscope.diagnostics.series import fmt, is_number, local_noise_sigma, robust_sigma, tokenize

# Settings that identify a run rather than configure it; they can't explain a difference.
IDENTITY_TOKENS = {'seed', 'name', 'id', 'dir', 'path', 'output', 'run', 'tag', 'notes', 'wandb'}


def comparable(ctx: RunContext) -> bool:
    """Crashed or interrupted runs stopped early; comparing their 'final' values to finished runs is misleading."""
    return (ctx.status() or {}).get('status') not in ('crashed', 'interrupted')


def _hashable(v) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _config_diff(ctx: RunContext, others: list[RunContext], limit: int = 3) -> str:
    """Settings where this run differs from a value all the other runs share."""
    cfg = ctx.config()
    other_cfgs = [o.config() for o in others if o.config()]
    if not cfg or not other_cfgs:
        return ""
    diffs = []
    for key, value in cfg.items():
        values = [c.get(key) for c in other_cfgs]
        if key in other_cfgs[0] and all(v == values[0] for v in values) and value != values[0]:
            diffs.append(f"{key}={value!r} (others {values[0]!r})")
    return f" Its config differs in: {', '.join(diffs[:limit])}." if diffs else ""


def _join(items: list[str], limit: int = 3) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


@cross_run_diagnostic_rule(name="outlier_runs", description="Runs whose final validation loss is far worse than the rest")
def check_outlier_runs(contexts: list[RunContext]):
    finals: dict[str, list[tuple[RunContext, float]]] = {}
    for ctx in filter(comparable, contexts):
        losses = [m for m in ctx.metrics_of_kind('loss') if m.split == 'val'] or ctx.primary_losses()
        for m in losses:
            t = ctx.trend(m.key)
            if t is not None:
                finals.setdefault(m.key, []).append((ctx, t.value(t.last)))

    results = []
    for key, entries in finals.items():
        if len(entries) < 3:
            continue
        values = [v for _, v in entries]
        median = statistics.median(values)
        limit = median + max(3 * robust_sigma(values), 0.5 * abs(median))
        for ctx, v in entries:
            if v > limit:
                commit = ctx.run.get('git_commit')
                others = [c for c, _ in entries if c is not ctx]
                results.append(DiagnosticResult(
                    'info', "Outlier run",
                    f"Final '{key}' is {fmt(v)}, far worse than the median of the {len(entries)} selected runs "
                    f"({fmt(median)}).{_config_diff(ctx, others)}",
                    ctx.name,
                    suggested_action="Compare its settings, seed and code version"
                                     + (f" (commit {commit})" if commit else "") + " with the better runs.",
                    metric=key,
                ))
    return results


@cross_run_diagnostic_rule(name="metric_name_mismatch", description="Same metric logged under different names")
def check_metric_name_mismatch(contexts: list[RunContext]):
    examples = []
    for i, a in enumerate(contexts):
        for b in contexts[i + 1:]:
            a_keys, b_keys = a.metrics(), b.metrics()
            for ka, ma in a_keys.items():
                if ka in b_keys or ma.kind == 'other':
                    continue
                for kb, mb in b_keys.items():
                    if kb not in a_keys and (ma.kind, ma.split, ma.base) == (mb.kind, mb.split, mb.base):
                        examples.append(f"'{ka}' ({a.name}) vs '{kb}' ({b.name})")
    if not examples:
        return None
    return DiagnosticResult(
        'info', "Runs use different metric names",
        f"These look like the same metric under different names, so they are drawn in separate charts: "
        f"{_join(examples)}.",
        f"{len(contexts)} runs",
        suggested_action="Use one naming scheme (e.g. 'loss/train', 'loss/val') across experiments.",
    )


@cross_run_diagnostic_rule(name="eval_set_mismatch", description="Confusion matrices computed on different-sized sets")
def check_eval_set_mismatch(contexts: list[RunContext]):
    totals = []
    for ctx in contexts:
        records = ctx.records(MATRIX_FILE)
        c = parse_confusion(records[-1])[0] if records else None
        if c is not None and c.total > 0:
            totals.append((ctx.name, c.total))
    if len(totals) < 2:
        return None
    lo, hi = min(t for _, t in totals), max(t for _, t in totals)
    if hi <= 1.01 * lo:
        return None
    return DiagnosticResult(
        'info', "Runs evaluated on different data",
        f"Latest confusion matrices contain between {lo:g} and {hi:g} samples "
        f"({_join([f'{name}: {t:g}' for name, t in totals])}), so their scores aren't directly comparable.",
        f"{len(totals)} runs",
        suggested_action="Evaluate every run on the same fixed validation set.",
    )


@cross_run_diagnostic_rule(name="uncommitted_code", description="Runs launched from a dirty git working tree")
def check_uncommitted_code(contexts: list[RunContext]):
    dirty = [ctx for ctx in contexts if str(ctx.run.get('git_commit', '')).endswith('*')]
    if not dirty:
        return None
    who = dirty[0].name if len(dirty) == 1 else f"{len(dirty)} of {len(contexts)} runs"
    return DiagnosticResult(
        'info', "Run from uncommitted code",
        "Launched with uncommitted changes (commit marked '*'), so the exact code can't be recovered "
        f"from git: {_join([ctx.name for ctx in dirty])}.",
        who,
        suggested_action="Commit before launching experiments you may need to reproduce.",
    )


def _comparison_metric(ctx: RunContext):
    """The metric a person would rank runs by: a val score, else a val loss, else any score/loss."""
    for kind, split in (('score', 'val'), ('loss', 'val'), ('score', None), ('loss', None)):
        for m in ctx.metrics_of_kind(kind):
            if m.split == split:
                return m
    return None


@cross_run_diagnostic_rule(name="inconclusive_comparison", description="Best and runner-up runs are within noise")
def check_inconclusive_comparison(contexts: list[RunContext]):
    groups: dict[str, list] = {}
    for ctx in filter(comparable, contexts):
        m = _comparison_metric(ctx)
        t = ctx.trend(m.key) if m else None
        if t is None or t.n < 5:
            continue
        tail = t.raw[-max(10, t.n // 3):]
        std_err = local_noise_sigma(tail) / math.sqrt(t.window_k)
        groups.setdefault(m.key, []).append((t.tail_mean(), std_err, ctx, t))
    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda e: e[0])  # oriented: lower is better for losses and (negated) scores
        (v1, se1, best, t), (v2, se2, second, _) = entries[0], entries[1]
        noise = 2 * math.sqrt(se1 ** 2 + se2 ** 2)
        if v2 - v1 >= noise or noise == 0:
            continue
        if v2 - v1 <= 1e-9 * max(1.0, abs(v1)):
            ranking = f"{best.name} and {second.name} are tied at {fmt(t.value(v1))}"
        else:
            ranking = (f"{best.name} ({fmt(t.value(v1))}) edges out {second.name} ({fmt(t.value(v2))}) "
                       f"by {fmt(v2 - v1)}, but epoch-to-epoch noise alone is about ±{fmt(noise)}")
        return DiagnosticResult(
            'info', "Top runs are within noise",
            f"By '{key}' (mean of the last epochs), {ranking}. "
            "Seed-to-seed variation is usually larger still, so the ranking may not hold.",
            f"{len(entries)} runs",
            suggested_action="Repeat the top runs with 2–3 seeds, or evaluate on a larger validation set, "
                             "before choosing between them.",
            metric=key,
        )
    return None


PROBLEM_SEVERITIES = {'critical', 'warning'}


@cross_run_diagnostic_rule(name="config_attribution", description="Training problems that line up with a setting")
def check_config_attribution(contexts: list[RunContext]):
    with_config = [c for c in contexts if c.config()]
    bad = [c for c in with_config if any(r.severity in PROBLEM_SEVERITIES for r in c.results)]
    good = [c for c in with_config if c not in bad]
    if len(bad) < 2 or len(good) < 2:
        return None

    findings = []
    keys = sorted(set().union(*(c.config().keys() for c in with_config)))
    for key in keys:
        if set(tokenize(key)) & IDENTITY_TOKENS:
            continue
        bad_vals = [c.config().get(key, None) for c in bad]
        good_vals = [c.config().get(key, None) for c in good]
        if not all(_hashable(v) for v in bad_vals + good_vals):
            continue
        if all(is_number(v) for v in bad_vals + good_vals):
            if min(bad_vals) > max(good_vals):
                findings.append(f"{key} ≥ {min(bad_vals):g} (healthy runs ≤ {max(good_vals):g})")
            elif max(bad_vals) < min(good_vals):
                findings.append(f"{key} ≤ {max(bad_vals):g} (healthy runs ≥ {min(good_vals):g})")
        elif not set(bad_vals) & set(good_vals):
            shown = ", ".join(sorted({repr(v) for v in bad_vals}))
            findings.append(f"{key} = {shown} (healthy runs: {', '.join(sorted({repr(v) for v in good_vals}))})")
    if not findings:
        return None
    names = ", ".join(c.name for c in bad[:3]) + (f" +{len(bad) - 3}" if len(bad) > 3 else "")
    return DiagnosticResult(
        'info', "Problems line up with a setting",
        f"The {len(bad)} runs with warnings ({names}) all have {'; '.join(findings[:3])}, unlike the "
        f"{len(good)} healthy runs.",
        f"{len(with_config)} runs",
        suggested_action="Treat this as a lead, not proof: with few runs, an unrelated setting can separate "
                         "them by chance. Change that one setting and re-run to confirm.",
    )


@cross_run_diagnostic_rule(name="config_consistency", description="Same setting stored with different types")
def check_config_consistency(contexts: list[RunContext]):
    examples = []
    configs = [(c, c.config()) for c in contexts if c.config()]
    keys = sorted(set().union(*(cfg.keys() for _, cfg in configs))) if configs else []
    for key in keys:
        typed = {}
        for ctx, cfg in configs:
            if key in cfg and cfg[key] is not None:
                kind = 'number' if is_number(cfg[key]) else type(cfg[key]).__name__
                typed.setdefault(kind, (ctx, cfg[key]))
        if 'number' in typed and 'str' in typed:
            (nctx, nval), (sctx, sval) = typed['number'], typed['str']
            examples.append(f"{key}: {nval!r} in {nctx.name} but the string {sval!r} in {sctx.name}")
    if not examples:
        return None
    return DiagnosticResult(
        'info', "Config values stored with different types",
        f"{'; '.join(examples[:3])}. Numbers passed as strings (often from command-line args) can silently "
        "behave differently.",
        f"{len(configs)} runs",
        suggested_action="Convert command-line arguments to their real types before using and logging them.",
    )
