"""Diagnostics for scalar curves in metrics.jsonl: data integrity, optimisation and generalisation."""
import math
import os
import statistics
import time
from datetime import datetime
from typing import Optional

from xscope.diagnostics.core import DiagnosticResult, RunContext, diagnostic_rule
from xscope.diagnostics.series import MetricInfo, fmt, format_x, is_number, noise_sigma

METRIC_FILES = ("metrics.jsonl", "2d.jsonl", "matrix.jsonl", "text.jsonl")


def _epoch(x) -> Optional[int]:
    if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and float(x).is_integer():
        return int(x)
    return None


def _result(ctx: RunContext, severity: str, title: str, message: str,
            action: str | None = None, x=None, metric: str | None = None) -> DiagnosticResult:
    return DiagnosticResult(
        severity=severity, title=title, message=message, run_name=ctx.name,
        epoch=_epoch(x), suggested_action=action, metric=metric,
    )


def _x_of(record: dict, index: int):
    return record.get('epoch', record.get('step', index + 1))


def _join(items: list[str], limit: int = 3) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


# ---------------------------------------------------------------------------
# Shared detectors (memoised per run so rules can consult each other)
# ---------------------------------------------------------------------------

def learning_progress(ctx: RunContext, m: MetricInfo) -> Optional[tuple[float, bool]]:
    """(ratio, significant): how far the end of the run is below its start.

    ratio is the relative drop for losses, or the share of remaining headroom for bounded
    scores. Start and end are averages of several raw points, and `significant` says the
    gain clearly exceeds what noise alone could produce.
    """
    def compute():
        t = ctx.trend(m.key)
        if t is None:
            return None
        start, end = t.head_mean(), t.tail_mean()
        gain = start - end
        significant = gain > 3 * t.sigma * math.sqrt(2 / t.window_k)
        # A model can converge within its first logged epoch; a clear drop from the very
        # first point still counts as learning.
        first_drop = t.raw[0] - end
        if first_drop > max(3 * t.sigma, 0.02 * abs(t.raw[0])) and first_drop > gain:
            start, gain, significant = t.raw[0], first_drop, True
        if m.kind == 'score' and m.bounded:
            _, ys = ctx.series(m.key)
            upper = 100.0 if max(ys) > 1 else 1.0
            headroom = upper - t.value(start)
            return (gain / headroom, significant) if headroom > 0 else None
        scale = abs(start)
        return (gain / scale, significant) if scale > 1e-12 else None
    return ctx.memo(('learning_progress', m.key), compute)


def is_learning(ctx: RunContext, m: MetricInfo) -> Optional[bool]:
    progress = learning_progress(ctx, m)
    if progress is None:
        return None
    ratio, significant = progress
    return ratio >= 0.02 and significant


def divergence(ctx: RunContext, key: str) -> Optional[dict]:
    """'critical' when a loss climbed back above its starting value, 'warning' when it gave back
    a large share of its improvement."""
    def compute():
        t = ctx.trend(key)
        if t is None or t.n < 5:
            return None
        rise = t.last - t.best
        points_after_best = len(t.smooth) - 1 - t.best_index
        if t.last > t.start + max(3 * t.sigma, 0.25 * abs(t.start)) and rise > 5 * t.sigma:
            return {'level': 'critical', 't': t}
        if (t.improvement > 0 and points_after_best >= 2
                and rise > max(5 * t.sigma, 0.3 * t.improvement)):
            return {'level': 'warning', 't': t}
        return None
    return ctx.memo(('divergence', key), compute)


def overfitting(ctx: RunContext, train: MetricInfo, val: MetricInfo) -> Optional[dict]:
    """Val loss rose meaningfully above its best while (usually) the train loss kept falling."""
    def compute():
        v, t = ctx.trend(val.key), ctx.trend(train.key)
        if v is None or t is None or v.n < 6 or is_learning(ctx, train) is False:
            return None  # without training progress a val rise isn't overfitting (not_learning covers it)
        region, k = v.best_region(), v.window_k
        if max(region) >= v.n - k:
            return None  # best is among the latest epochs: nothing has risen yet
        best_mean = statistics.fmean(v.raw[i] for i in region)
        now = v.tail_mean()
        rise = now - best_mean
        std_err = v.sigma * math.sqrt(1 / k + 1 / len(region))
        if rise <= max(3.5 * std_err, 0.05 * v.improvement, 0.02 * abs(best_mean), 1e-12):
            return None
        train_at_best = t.value_near(v.best_x)
        train_now = t.tail_mean()
        train_improving = train_at_best - train_now > max(2 * t.sigma / math.sqrt(t.window_k),
                                                          0.01 * abs(train_at_best))
        return {'v': v, 't': t, 'now': now, 'train_at_best': train_at_best, 'train_now': train_now,
                'train_improving': train_improving}
    return ctx.memo(('overfitting', train.key, val.key), compute)


def any_overfitting(ctx: RunContext) -> list[tuple[MetricInfo, MetricInfo, dict]]:
    found = []
    for train, val in ctx.pairs('loss'):
        o = overfitting(ctx, train, val)
        if o:
            found.append((train, val, o))
    return found


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

@diagnostic_rule(name="nonfinite_values", description="NaN or Inf in any logged metric")
def check_nonfinite_values(ctx: RunContext):
    first: dict[str, tuple] = {}
    for i, r in enumerate(ctx.records()):
        for k, v in r.items():
            if is_number(v) and not math.isfinite(v) and k not in first:
                first[k] = (_x_of(r, i), v)
    if not first:
        return None
    ordered = sorted(first.items(), key=lambda kv: kv[1][0] if is_number(kv[1][0]) else 0)
    parts = [f"'{k}' ({v}) at epoch {format_x(x)}" for k, (x, v) in ordered]
    return _result(
        ctx, 'critical', "Non-finite metric values",
        f"NaN/Inf first appeared in {_join(parts)}.",
        "Lower the learning rate, enable gradient clipping, and look for log(0), division by zero "
        "or corrupt samples in the batches around that epoch.",
        x=ordered[0][1][0], metric=ordered[0][0],
    )


@diagnostic_rule(name="metric_integrity", description="Restarted epochs, frozen, out-of-range or vanished metrics")
def check_metric_integrity(ctx: RunContext):
    records = ctx.records()
    results = []

    xs = [x for x in (r.get('epoch', r.get('step')) for r in records) if is_number(x)]
    backwards = [(a, b) for a, b in zip(xs, xs[1:]) if b < a]
    if backwards:
        a, b = backwards[0]
        results.append(_result(
            ctx, 'warning', "Epoch counter went backwards",
            f"The epoch value decreased {len(backwards)} time(s) (first {format_x(a)} → {format_x(b)}), "
            "so this file mixes several training sessions and its curves overlap.",
            "Create one Tracker per training session or fold, or log a monotonically increasing step.",
        ))

    frozen, out_of_range, negative = [], [], []
    for m in ctx.metrics().values():
        _, ys = ctx.series(m.key)
        if not ys:
            continue
        lo, hi = min(ys), max(ys)
        if m.kind in ('loss', 'score', 'grad') and len(ys) >= 8 and lo == hi:
            frozen.append(f"'{m.key}' = {fmt(lo)}")
        if m.bounded and (lo < 0 or hi > 100):
            out_of_range.append(f"'{m.key}' spans [{fmt(lo)}, {fmt(hi)}]")
        if m.non_negative and lo < -1e-9:
            negative.append(f"'{m.key}' reaches {fmt(lo)}")

    if frozen:
        results.append(_result(
            ctx, 'warning', "Metric never changes",
            f"Identical value in every record: {_join(frozen)}.",
            "Make sure the value is recomputed each epoch (not a stale variable) and that the "
            "optimizer is actually updating the model's parameters.",
        ))
    if out_of_range:
        results.append(_result(
            ctx, 'warning', "Metric out of valid range",
            f"Expected within [0, 1] (or [0, 100] as a percentage): {_join(out_of_range)}.",
            "Check how the metric is computed and averaged (e.g. summing instead of averaging over batches).",
        ))
    if negative:
        results.append(_result(
            ctx, 'warning', "Impossible negative loss",
            f"These losses can't be negative by definition: {_join(negative)}.",
            "Check the sign of the loss and any terms subtracted from it (regularizers, custom label smoothing).",
        ))

    if len(records) >= 10:
        vanished = []
        for m in ctx.metrics().values():
            idx = [i for i, r in enumerate(records) if is_number(r.get(m.key))]
            if len(idx) < 3:
                continue
            typical_gap = statistics.median(b - a for a, b in zip(idx, idx[1:]))
            missing_tail = len(records) - 1 - idx[-1]
            if missing_tail > max(5, 3 * typical_gap):
                vanished.append(f"'{m.key}' (last at epoch {format_x(_x_of(records[idx[-1]], idx[-1]))})")
        if vanished:
            results.append(_result(
                ctx, 'info', "Metric stopped being logged",
                f"No longer present in recent records: {_join(vanished)}.",
                "If unintended, check the code path that logs it (e.g. an evaluation step that stopped running).",
            ))
    return results


@diagnostic_rule(name="empty_run", description="Run directory without any logged data")
def check_empty_run(ctx: RunContext):
    if any(ctx.records(f) for f in METRIC_FILES):
        return None
    return _result(
        ctx, 'info', "No data logged",
        "This run has no metrics, 2D or matrix records.",
        "The process probably crashed before its first log() call, or logged to a different directory.",
    )


# ---------------------------------------------------------------------------
# Optimisation dynamics
# ---------------------------------------------------------------------------

@diagnostic_rule(name="loss_divergence", description="Loss blew up or gave back much of its progress")
def check_loss_divergence(ctx: RunContext):
    results = []
    for m in ctx.primary_losses():
        d = divergence(ctx, m.key)
        if not d:
            continue
        t = d['t']
        best_x, best = t.checkpoint()
        if d['level'] == 'critical':
            results.append(_result(
                ctx, 'critical', "Loss diverged",
                f"'{m.key}' is now {fmt(t.value(t.last))}, above its starting value {fmt(t.value(t.start))} "
                f"(best was {fmt(t.value(best))} at epoch {format_x(best_x)}).",
                "Training has blown up: lower the learning rate, add warmup or gradient clipping, "
                f"and restore the checkpoint from epoch {format_x(best_x)}.",
                x=best_x, metric=m.key,
            ))
        else:
            lost = 100 * (t.last - t.best) / t.improvement
            results.append(_result(
                ctx, 'warning', "Loss rising",
                f"'{m.key}' climbed from its best {fmt(t.value(best))} (epoch {format_x(best_x)}) to "
                f"{fmt(t.value(t.last))}, giving back {lost:.0f}% of its improvement.",
                "A training loss shouldn't rise for long under a stable optimizer: lower the LR, check for "
                "LR-schedule restarts, or look for a change in the data pipeline.",
                x=best_x, metric=m.key,
            ))
    return results


def _lr_values(ctx: RunContext) -> tuple[list, list]:
    for lr in ctx.metrics_of_kind('lr'):
        xs, ys = ctx.series(lr.key)
        if len(ys) >= 3:
            return xs, ys
    return [], []


def _starts_at_peak_lr(ctx: RunContext) -> bool:
    _, ys = _lr_values(ctx)
    return bool(ys) and max(ys) > 0 and ys[0] >= 0.9 * max(ys)


def _lr_change_note(ctx: RunContext, x_before, x_at) -> str:
    for lr in ctx.metrics_of_kind('lr'):
        lxs, lys = ctx.series(lr.key)
        before = [y for x, y in zip(lxs, lys) if x <= x_before]
        at = [y for x, y in zip(lxs, lys) if x <= x_at]
        if before and at and before[-1] > 0 and at[-1] > 1.1 * before[-1]:
            return f" It coincides with a learning-rate increase ({fmt(before[-1])} → {fmt(at[-1])})."
    return ""


@diagnostic_rule(name="loss_spikes", description="Sudden jumps in a loss curve")
def check_loss_spikes(ctx: RunContext):
    results = []
    for m in ctx.primary_losses():
        t = ctx.trend(m.key)
        if t is None or t.n < 10:
            continue
        d = divergence(ctx, m.key)
        if d and d['level'] == 'critical':
            continue
        ys, n = t.raw, t.n
        spike_idx = []
        for i in range(5, n):
            pre = statistics.median(ys[i - 5:i])
            if ys[i] - pre > max(8 * t.sigma, 0.3 * abs(pre), 1e-12):
                spike_idx.append(i)
        if not spike_idx:
            continue
        events = [i for i in spike_idx if i - 1 not in spike_idx]
        i = events[-1]
        pre = statistics.median(ys[i - 5:i])
        peak = max(ys[i:i + 3])
        recovered = any(y <= pre + max(3 * t.sigma, 0.1 * abs(pre)) for y in ys[i + 1:])
        count = f" ({len(events)} spikes in total)" if len(events) > 1 else ""
        note = _lr_change_note(ctx, t.xs[i - 1], t.xs[i])
        if i <= 0.1 * n and _starts_at_peak_lr(ctx):
            note += " The run starts at its full learning rate; a short warmup usually prevents early spikes."
        if recovered:
            results.append(_result(
                ctx, 'info', "Loss spike (recovered)",
                f"'{m.key}' jumped from ~{fmt(pre)} to {fmt(peak)} at epoch {format_x(t.xs[i])}{count} "
                f"and later recovered.{note}",
                "Isolated spikes usually come from bad batches or LR restarts; if they recur, add gradient "
                "clipping or lower the learning rate.",
                x=t.xs[i], metric=m.key,
            ))
        else:
            results.append(_result(
                ctx, 'warning', "Loss spike",
                f"'{m.key}' jumped from ~{fmt(pre)} to {fmt(peak)} at epoch {format_x(t.xs[i])}{count} "
                f"and hasn't recovered yet.{note}",
                "Lower the learning rate or add gradient clipping, and inspect the data seen around that epoch.",
                x=t.xs[i], metric=m.key,
            ))
    return results


@diagnostic_rule(name="training_instability", description="Loss noise growing as training progresses")
def check_training_instability(ctx: RunContext):
    results = []
    for m in ctx.primary_losses():
        t = ctx.trend(m.key)
        if t is None or t.n < 45:  # each third needs ~15 points for a stable noise estimate
            continue
        d = divergence(ctx, m.key)
        if d and d['level'] == 'critical':
            continue
        third = t.n // 3
        early, late = t.raw[:third], t.raw[-third:]
        s_early, s_late = noise_sigma(early), noise_sigma(late)
        l_early, l_late = abs(statistics.median(early)), abs(statistics.median(late))
        if l_early <= 0 or l_late <= 0:
            continue
        rel_early, rel_late = s_early / l_early, s_late / l_late
        if s_late > 2 * s_early and rel_late > max(2.5 * rel_early, 0.05):
            growth = f"{s_late / s_early:.1f}×" if s_early > 0 else "from ~0"
            results.append(_result(
                ctx, 'warning', "Training becoming unstable",
                f"Epoch-to-epoch noise in '{m.key}' grew {growth} between the first and last third of "
                f"training and is now ±{100 * rel_late:.0f}% of the loss level.",
                "Usually a learning rate that is too high for the later phase: add LR decay, lower the LR, "
                "or enable gradient clipping.",
                metric=m.key,
            ))
    return results


@diagnostic_rule(name="not_learning", description="Loss (or score) barely moved")
def check_not_learning(ctx: RunContext):
    candidates = ctx.primary_losses()
    if not candidates:
        candidates = ctx.metrics_of_kind('score')
    results = []
    for m in candidates:
        t = ctx.trend(m.key)
        if t is None or t.n < 8 or divergence(ctx, m.key):
            continue
        if is_learning(ctx, m) is not False:
            continue
        ratio, _ = learning_progress(ctx, m)
        lr_note = ""
        for lr in ctx.metrics_of_kind('lr'):
            _, lys = ctx.series(lr.key)
            if lys:
                lr_note = f" Current learning rate: {fmt(lys[-1])}."
                break
        progress = "of its remaining headroom" if m.kind == 'score' and m.bounded else "relative change"
        results.append(_result(
            ctx, 'warning', "Not learning",
            f"'{m.key}' averaged {fmt(t.value(t.head_mean()))} over its first {t.window_k} epochs and "
            f"{fmt(t.value(t.tail_mean()))} over its last {t.window_k} ({100 * ratio:.1f}% {progress}, "
            f"within noise or under 2%).{lr_note}",
            "Try learning rates 10× higher and lower, check that the optimizer receives the model's parameters, "
            "that layers aren't frozen by accident, and that labels line up with inputs. Overfitting a single "
            "batch is a quick sanity test.",
            metric=m.key,
        ))
    return results


def _no_decay_note(ctx: RunContext, x_start) -> str:
    lxs, lys = _lr_values(ctx)
    during = [y for x, y in zip(lxs, lys) if x >= x_start]
    if len(during) >= 3 and max(during) > 0 and min(during) >= 0.9 * max(during):
        return " The learning rate was never decayed during the plateau; a step or cosine decay often breaks it."
    return ""


@diagnostic_rule(name="plateau", description="Loss stopped improving (converged)")
def check_plateau(ctx: RunContext):
    overfit_keys = {k for train, val, _ in any_overfitting(ctx) for k in (train.key, val.key)}
    results = []
    for m in ctx.primary_losses():
        t = ctx.trend(m.key)
        if t is None or t.n < 12 or m.key in overfit_keys or divergence(ctx, m.key):
            continue
        if not is_learning(ctx, m) or t.improvement <= 0:
            continue
        tol = max(0.02 * t.improvement, t.sigma)
        S = t.smooth
        start = len(S)
        while start > 0 and S[start - 1] - t.best <= tol:
            start -= 1
        if start == 0 or start >= len(S):
            continue
        x_start = t.smooth_xs[start]
        epochs_flat = sum(1 for x in t.xs if x >= x_start)
        if epochs_flat < max(5, 0.25 * t.n):
            continue
        results.append(_result(
            ctx, 'info', "Loss has plateaued",
            f"'{m.key}' has stayed within 2% of its total improvement since epoch {format_x(x_start)} "
            f"({epochs_flat} of {t.n} epochs).{_no_decay_note(ctx, x_start)}",
            "More epochs at this learning rate are unlikely to help: stop earlier to save compute, "
            "or decay the learning rate to squeeze out further gains.",
            x=x_start, metric=m.key,
        ))
    return results


@diagnostic_rule(name="gradient_health", description="Exploding or vanishing gradient norms")
def check_gradient_health(ctx: RunContext):
    results = []
    for m in ctx.metrics_of_kind('grad'):
        xs, ys = ctx.series(m.key)
        if len(ys) < 10:
            continue
        early = statistics.median(ys[:max(3, len(ys) // 3)])
        recent = statistics.median(ys[-3:])
        if early <= 0:
            continue
        if recent > 10 * early:
            results.append(_result(
                ctx, 'critical', "Exploding gradients",
                f"'{m.key}' is now {fmt(recent)}, {recent / early:.0f}× its early level ({fmt(early)}).",
                "Enable gradient clipping and lower the learning rate; check normalization layers.",
                metric=m.key,
            ))
        elif max(ys) > 100 * early:
            i = ys.index(max(ys))
            results.append(_result(
                ctx, 'warning', "Gradient spike",
                f"'{m.key}' peaked at {fmt(ys[i])} at epoch {format_x(xs[i])}, {ys[i] / early:.0f}× its early level.",
                "Gradient clipping protects against rare huge updates.",
                x=xs[i], metric=m.key,
            ))
        elif recent < 1e-3 * early:
            results.append(_result(
                ctx, 'warning', "Vanishing gradients",
                f"'{m.key}' fell from {fmt(early)} to {fmt(recent)} (1000× smaller).",
                "Check for saturated activations, dead ReLUs, or a learning-rate schedule that reached zero.",
                metric=m.key,
            ))
    return results


# ---------------------------------------------------------------------------
# Generalisation
# ---------------------------------------------------------------------------

def _val_score_note(ctx: RunContext, best_x) -> str:
    notes = []
    for s in ctx.metrics_of_kind('score'):
        if s.split != 'val':
            continue
        t = ctx.trend(s.key)
        if t is None:
            continue
        at_best, now = t.value(t.value_near(best_x)), t.value(t.last)
        if now >= at_best - 2 * t.sigma:
            notes.append(f" '{s.key}' has not dropped ({fmt(at_best)} → {fmt(now)}), so the model is mostly "
                         "becoming over-confident (worse calibration) rather than less accurate.")
        else:
            notes.append(f" '{s.key}' also dropped ({fmt(at_best)} → {fmt(now)}).")
    return "".join(notes[:1])


@diagnostic_rule(name="overfitting", description="Validation loss rising after its best epoch")
def check_overfitting(ctx: RunContext):
    results = []
    for train, val, o in any_overfitting(ctx):
        v, t = o['v'], o['t']
        best_x, best = v.checkpoint()
        now = o['now']
        pct = 100 * (now - best) / abs(best) if best else float('inf')
        note = _val_score_note(ctx, best_x)
        if o['train_improving']:
            results.append(_result(
                ctx, 'warning', "Overfitting",
                f"'{val.key}' was best at epoch {format_x(best_x)} ({fmt(v.value(best))}) and is now "
                f"~{fmt(v.value(now))} (+{pct:.0f}%), while '{train.key}' kept falling "
                f"({fmt(t.value(o['train_at_best']))} → {fmt(t.value(o['train_now']))}).{note}",
                f"Early-stop or keep the checkpoint from epoch {format_x(best_x)}; add regularization "
                "(weight decay, dropout, augmentation) or more training data.",
                x=best_x, metric=val.key,
            ))
        else:
            results.append(_result(
                ctx, 'warning', "Validation loss regressing",
                f"'{val.key}' rose from {fmt(v.value(best))} (epoch {format_x(best_x)}) to "
                f"~{fmt(v.value(now))} (+{pct:.0f}%) while '{train.key}' was not improving.{note}",
                f"Keep the checkpoint from epoch {format_x(best_x)}; check for LR-schedule problems or a "
                "mismatch between training and validation data.",
                x=best_x, metric=val.key,
            ))
    return results


@diagnostic_rule(name="val_below_train", description="Validation loss consistently below training loss")
def check_val_below_train(ctx: RunContext):
    results = []
    overfit_vals = {val.key for _, val, _ in any_overfitting(ctx)}
    for train, val in ctx.pairs('loss'):
        if val.key in overfit_vals:
            continue
        tr = dict(zip(*ctx.series(train.key)))
        va = dict(zip(*ctx.series(val.key)))
        common = sorted(set(tr) & set(va))
        if len(common) < 5:
            continue
        last5 = common[-5:]
        below = sum(1 for x in last5 if va[x] < tr[x] - 0.05 * abs(tr[x]))
        if below >= 4:
            x = last5[-1]
            results.append(_result(
                ctx, 'info', "Validation loss below training loss",
                f"In {below} of the last 5 epochs '{val.key}' was more than 5% below '{train.key}' "
                f"(latest {fmt(va[x])} vs {fmt(tr[x])}).",
                "Usually benign: dropout, augmentation and label smoothing only affect the train loss, and the "
                "train loss is averaged over an epoch in which the model was still improving. If you use none of "
                "these, check for overlap between training and validation data.",
                metric=val.key,
            ))
    return results


@diagnostic_rule(name="score_regression", description="Accuracy-like metric dropped from its best")
def check_score_regression(ctx: RunContext):
    val_overfit = bool(any_overfitting(ctx))
    results = []
    for m in ctx.metrics_of_kind('score'):
        if m.split == 'val' and val_overfit:
            continue  # already explained in the overfitting finding
        t = ctx.trend(m.key)
        if t is None or t.n < 5 or t.improvement <= 0:
            continue
        if t.best_index >= len(t.smooth) - 1:
            continue
        current = statistics.median(t.raw[-3:])
        drop = current - t.best
        best_x, best = t.checkpoint()
        if drop > max(4 * t.sigma, 0.15 * t.improvement, 1e-12):
            lost = 100 * drop / t.improvement
            results.append(_result(
                ctx, 'warning', "Metric regressed",
                f"'{m.key}' peaked at {fmt(t.value(best))} (epoch {format_x(best_x)}) and is now "
                f"{fmt(t.value(current))}, losing {min(lost, 999):.0f}% of its gains.",
                f"Use the checkpoint from epoch {format_x(best_x)}. A sudden drop points to an LR restart or "
                "data change; a gradual one on validation data is overfitting.",
                x=best_x, metric=m.key,
            ))
    return results


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.0f} days"


@diagnostic_rule(name="stopped_while_improving", description="Run ended while its loss was still falling")
def check_stopped_while_improving(ctx: RunContext):
    """The last 10% of epochs still produced > 5% of the total improvement when the run ended.

    Using only the final 10% keeps cosine/step schedules quiet: their learning rate is ~0 at
    the very end, so the curve flattens even though the previous decay phase gained a lot.
    """
    status = (ctx.status() or {}).get('status')
    if status in ('crashed', 'interrupted', 'running'):
        return None  # run_status covers these
    n = len(ctx.records())
    if n < 10:
        return None

    if status != 'finished':
        # Older runs without status.json: infer "ended" from how long the file has been quiet.
        path = os.path.join(ctx.run_path, "metrics.jsonl")
        if not os.path.isfile(path):
            return None
        times = ctx.times()
        if len(times) >= 2:
            started, last_write = times[0], times[-1]
        else:
            try:
                started = datetime.strptime(str(ctx.run.get('timestamp', '')), "%Y%m%d_%H%M%S").timestamp()
            except ValueError:
                return None
            last_write = os.path.getmtime(path)
        if last_write <= started:
            return None
        cadence = (last_write - started) / n
        if time.time() - last_write < max(30 * cadence, 900):
            return None

    for m in ctx.primary_losses():
        t = ctx.trend(m.key)
        if t is None or t.n < 10 or t.improvement <= 0 or divergence(ctx, m.key):
            continue
        # Least-squares slope of the raw points in the final 10% (rolling medians flatten the ends).
        k = max(3, t.n // 10)
        idx = list(range(k))
        slope = statistics.linear_regression(idx, t.raw[-k:]).slope
        spread = math.sqrt(sum((i - (k - 1) / 2) ** 2 for i in idx))
        late_gain = -slope * k
        if late_gain > 0.05 * t.improvement and -slope > 2 * t.sigma / spread:
            ended = "finished" if status == 'finished' else "stopped logging"
            return _result(
                ctx, 'info', "Run ended while still improving",
                f"The run {ended} while '{m.key}' was still falling: the last 10% of epochs produced "
                f"{100 * late_gain / t.improvement:.0f}% of its total improvement.",
                "It was likely under-trained: train longer, or decay the learning rate to finish the descent.",
                metric=m.key,
            )
    return None


@diagnostic_rule(name="score_gap", description="Large train/validation gap on accuracy-like metrics")
def check_score_gap(ctx: RunContext):
    loss_overfit = bool(any_overfitting(ctx))
    results = []
    for train, val in ctx.pairs('score'):
        tt, vt = ctx.trend(train.key), ctx.trend(val.key)
        if tt is None or vt is None or tt.n < 5 or vt.n < 5:
            continue
        tr, va = tt.value(tt.tail_mean()), vt.value(vt.tail_mean())
        gap = tr - va
        if train.bounded:
            unit = 100.0 if max(ctx.series(train.key)[1]) > 1 else 1.0
            if gap <= 0.1 * unit:
                continue
            saturated = tr >= 0.995 * unit
        else:
            if gap <= 0.1 * abs(tr):
                continue
            saturated = False
        note = " Training accuracy is saturated: the model has essentially memorized the training set." if saturated else ""
        results.append(_result(
            ctx, 'info' if loss_overfit else 'warning', "Large train/validation gap",
            f"'{train.key}' averages {fmt(tr)} over the last {tt.window_k} epochs but '{val.key}' only "
            f"{fmt(va)} (gap {fmt(gap)}).{note}",
            "Add data or augmentation, strengthen regularization (weight decay, dropout), or use a smaller model; "
            "also check that train and validation data come from the same distribution.",
            metric=val.key,
        ))
    return results
