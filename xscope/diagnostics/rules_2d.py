"""Diagnostics for 2D point sets in 2d.jsonl, pairing ground-truth and prediction series by name."""
import math
import statistics
from typing import Optional

from xscope.diagnostics.core import DiagnosticResult, RunContext, diagnostic_rule
from xscope.diagnostics.series import X_AXIS_KEYS, fmt, format_x, is_finite_number, is_number, tokenize

FILE_2D = "2d.jsonl"
TRUE_TOKENS = {'true', 'gt', 'target', 'targets', 'label', 'labels', 'actual', 'groundtruth',
               'truth', 'ref', 'reference', 'expected'}
PRED_TOKENS = {'pred', 'preds', 'prediction', 'predictions', 'predicted', 'output', 'outputs',
               'est', 'estimate', 'estimated', 'yhat', 'hat'}


def point_keys(record: dict) -> list[str]:
    return [k for k in record if k.lower() not in X_AXIS_KEYS and not k.startswith('_')
            and isinstance(record[k], list)]


def pair_keys(keys: list[str]) -> list[tuple[str, str, str]]:
    """[(name, true_key, pred_key)] for series that look like ground truth vs prediction."""
    groups: dict[str, dict[str, str]] = {}
    for k in keys:
        tokens = tokenize(k)
        role = 'true' if set(tokens) & TRUE_TOKENS else 'pred' if set(tokens) & PRED_TOKENS else None
        if role:
            base = '/'.join(t for t in tokens if t not in TRUE_TOKENS | PRED_TOKENS) or '2d'
            groups.setdefault(base, {}).setdefault(role, k)
    return [(base, g['true'], g['pred']) for base, g in groups.items() if 'true' in g and 'pred' in g]


def valid_points(points) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in points
            if isinstance(p, list) and len(p) >= 2 and is_finite_number(p[0]) and is_finite_number(p[1])]


def valid_points_aligned(points: list) -> list[Optional[tuple[float, float]]]:
    """Like valid_points, but keeps positions (None for invalid points) so pairs stay aligned."""
    return [(float(p[0]), float(p[1]))
            if isinstance(p, list) and len(p) >= 2 and is_finite_number(p[0]) and is_finite_number(p[1]) else None
            for p in points]


def spread(pts: list[tuple[float, float]]) -> float:
    """Root of the summed per-axis variance: a scale for 'how big' the point cloud is."""
    if len(pts) < 2:
        return 0.0
    return math.sqrt(statistics.pvariance([p[0] for p in pts]) + statistics.pvariance([p[1] for p in pts]))


def _pair_points(record: dict, true_key: str, pred_key: str) -> Optional[tuple[list, list]]:
    t, p = record.get(true_key), record.get(pred_key)
    if not isinstance(t, list) or not isinstance(p, list) or len(t) != len(p):
        return None
    pairs = [(a, b) for a, b in zip(valid_points_aligned(t), valid_points_aligned(p)) if a and b]
    if not pairs:
        return None
    return [a for a, _ in pairs], [b for _, b in pairs]


def rmse(true_pts, pred_pts) -> float:
    return math.sqrt(sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for a, b in zip(true_pts, pred_pts)) / len(true_pts))


def _result(ctx: RunContext, severity: str, title: str, message: str, action: str | None,
            record: Optional[dict] = None, metric: Optional[str] = None) -> DiagnosticResult:
    x = record.get('epoch', record.get('step')) if record else None
    epoch = int(x) if is_finite_number(x) and float(x).is_integer() else None
    return DiagnosticResult(severity, title, message, ctx.name, epoch=epoch,
                            suggested_action=action, metric=metric)


@diagnostic_rule(name="points_2d_integrity", description="NaN/Inf, malformed points, mismatched pair lengths")
def check_points_2d_integrity(ctx: RunContext):
    records = ctx.records(FILE_2D)
    if not records:
        return None
    latest = records[-1]
    results = []
    nonfinite, malformed = [], []
    for k in point_keys(latest):
        pts = latest[k]
        bad_num = sum(1 for p in pts if isinstance(p, list) and len(p) >= 2
                      and any(is_number(v) and not math.isfinite(v) for v in p[:2]))
        bad_shape = sum(1 for p in pts if not (isinstance(p, list) and len(p) >= 2
                                               and all(is_number(v) for v in p[:2])))
        if bad_num:
            nonfinite.append(f"'{k}' ({bad_num} of {len(pts)})")
        if bad_shape:
            malformed.append(f"'{k}' ({bad_shape} of {len(pts)})")
    if nonfinite:
        results.append(_result(
            ctx, 'critical', "NaN/Inf in 2D points",
            f"Non-finite coordinates in {', '.join(nonfinite)}.",
            "The model's outputs are numerically unstable; lower the learning rate or check normalization.",
            latest,
        ))
    if malformed:
        results.append(_result(
            ctx, 'warning', "Malformed 2D points",
            f"Points that aren't [x, y] number pairs in {', '.join(malformed)}; they are skipped.",
            "Log each series as a list of [x, y] pairs.", latest,
        ))
    for name, tk, pk in pair_keys(point_keys(latest)):
        if len(latest[tk]) != len(latest[pk]):
            results.append(_result(
                ctx, 'info', "2D truth/prediction length mismatch",
                f"'{tk}' has {len(latest[tk])} points but '{pk}' has {len(latest[pk])}, "
                "so point-wise error can't be computed.",
                "Log predictions for exactly the same points as the ground truth, in the same order.",
                latest, metric=name,
            ))
    return results


@diagnostic_rule(name="prediction_2d_quality", description="Collapse, over-dispersion, offsets and outliers in 2D")
def check_prediction_2d_quality(ctx: RunContext):
    records = ctx.records(FILE_2D)
    if not records:
        return None
    latest = records[-1]
    results = []
    for name, tk, pk in pair_keys(point_keys(latest)):
        aligned = _pair_points(latest, tk, pk)
        if aligned is None or len(aligned[0]) < 2:
            continue
        T, P = aligned
        errors = [math.dist(a, b) for a, b in zip(T, P)]
        s_true = spread(T)
        err = rmse(T, P)

        # Outliers first, so a few wild points don't masquerade as global over-dispersion.
        med = statistics.median(errors)
        cutoff = max(10 * med, 0.5 * s_true)
        outliers = [i for i, e in enumerate(errors) if e > cutoff] if cutoff > 0 and len(errors) >= 5 else []
        if outliers and len(outliers) <= 0.2 * len(errors):
            worst = max(outliers, key=lambda i: errors[i])
            results.append(_result(
                ctx, 'info', "2D outlier predictions",
                f"{len(outliers)} of {len(errors)} points in '{pk}' are far off (>10× the median error); "
                f"worst is point {worst} with error {fmt(errors[worst])} vs median {fmt(med)}.",
                "Inspect those samples: they are often edge cases, occlusions or label errors.",
                latest, metric=name,
            ))
            keep = [i for i in range(len(errors)) if i not in set(outliers)]
            T, P = [T[i] for i in keep], [P[i] for i in keep]
            err = rmse(T, P)

        s_pred = spread(P)
        if s_true > 0:
            ratio = s_pred / s_true
            if ratio < 0.2:
                results.append(_result(
                    ctx, 'warning', "2D predictions collapsed",
                    f"'{pk}' spreads over only {ratio:.0%} of the area covered by '{tk}', i.e. the model predicts "
                    "nearly the same point for every input (regression to the mean).",
                    "The model may be ignoring its inputs: check feature normalization, the loss scale, and that "
                    "the target isn't normalized differently from the prediction.", latest, metric=name,
                ))
                continue
            if ratio > 3:
                results.append(_result(
                    ctx, 'warning', "2D predictions over-dispersed",
                    f"'{pk}' is spread {fmt(ratio)}× wider than '{tk}'.",
                    "Usually a scale mismatch: predictions in pixels vs normalized targets (or similar), "
                    "or a missing de-normalization step.", latest, metric=name,
                ))
                continue

        mean_dx = statistics.fmean(b[0] - a[0] for a, b in zip(T, P))
        mean_dy = statistics.fmean(b[1] - a[1] for a, b in zip(T, P))
        bias = math.hypot(mean_dx, mean_dy)
        if err > 0 and err > 0.05 * s_true and bias >= 0.7 * err:
            results.append(_result(
                ctx, 'info', "Systematic 2D offset",
                f"'{pk}' is shifted by ({fmt(mean_dx)}, {fmt(mean_dy)}) on average relative to '{tk}', "
                f"which accounts for most of its error (RMSE {fmt(err)}).",
                "Look for a coordinate-frame or origin mismatch, or a missing bias term — a constant "
                "offset is easy to correct.", latest, metric=name,
            ))

        if err <= 0.05 * s_true:
            continue  # already close to the truth; structure in tiny errors isn't actionable
        results.extend(_error_structure(ctx, latest, name, tk, pk, T, P))
    return results


def _error_structure(ctx: RunContext, record: dict, name: str, tk: str, pk: str, T: list, P: list):
    """Drift along the sequence, error concentrated on one axis, and per-axis scale bias."""
    results = []
    errors = [math.dist(a, b) for a, b in zip(T, P)]
    n = len(errors)

    if n >= 8 and statistics.pvariance(errors) > 0:
        q = n // 4
        first, last = statistics.fmean(errors[:q]), statistics.fmean(errors[-q:])
        corr = statistics.correlation(range(n), errors)
        if corr >= 0.6 and last >= 2.5 * max(first, 1e-12):
            results.append(_result(
                ctx, 'info', "2D error accumulates along the sequence",
                f"Error in '{pk}' grows from ~{fmt(first)} over the first points to ~{fmt(last)} over the last "
                f"(correlation with point index {corr:.2f}), typical of drift in step-by-step predictions.",
                "Train with multi-step losses or scheduled sampling, or re-anchor predictions periodically.",
                record, metric=name,
            ))

    spread_t = [statistics.pstdev([p[i] for p in T]) for i in (0, 1)]
    axis_rmse = [math.sqrt(statistics.fmean((b[i] - a[i]) ** 2 for a, b in zip(T, P))) for i in (0, 1)]
    if all(s > 0 for s in spread_t):
        rel = [axis_rmse[i] / spread_t[i] for i in (0, 1)]
        hi = 0 if rel[0] >= rel[1] else 1
        if rel[hi] > 0.05 and rel[hi] >= 3 * max(rel[1 - hi], 1e-12):
            results.append(_result(
                ctx, 'info', "2D error concentrated on one axis",
                f"Relative to the spread of '{tk}', the error of '{pk}' along {'xy'[hi]} is "
                f"{rel[hi] / max(rel[1 - hi], 1e-12):.1f}× the error along {'xy'[1 - hi]}.",
                "Check that axis specifically: its normalization, its share of the loss, or ambiguity in the "
                "inputs along that direction.", record, metric=name,
            ))

    scaled = []
    for i in (0, 1):
        tv, pv = [p[i] for p in T], [p[i] for p in P]
        if statistics.pstdev(tv) == 0 or statistics.pstdev(pv) == 0:
            continue
        slope = statistics.linear_regression(tv, pv).slope
        if statistics.correlation(tv, pv) >= 0.9 and 0.2 <= slope <= 3 and not 0.85 <= slope <= 1.18:
            scaled.append(f"{'xy'[i]} ≈ {slope:.2f}×")
    if scaled:
        results.append(_result(
            ctx, 'info', "2D predictions scaled relative to the truth",
            f"Around their center, '{pk}' is a scaled copy of '{tk}' ({', '.join(scaled)}).",
            "A factor below 1 is regression to the mean (the model hedges toward the average); a factor above 1, "
            "or different factors per axis, usually means a unit or normalization mismatch.", record, metric=name,
        ))
    return results


@diagnostic_rule(name="prediction_2d_trend", description="2D prediction error growing or not improving over epochs")
def check_prediction_2d_trend(ctx: RunContext):
    records = ctx.records(FILE_2D)
    if len(records) < 3:
        return None
    results = []
    for name, tk, pk in pair_keys(point_keys(records[-1])):
        history = []
        for i, r in enumerate(records):
            aligned = _pair_points(r, tk, pk)
            if aligned:
                history.append((r.get('epoch', r.get('step', i + 1)), rmse(*aligned), spread(aligned[0])))
        if len(history) < 3:
            continue
        b = min(range(len(history)), key=lambda i: history[i][1])
        best_x, best, _ = history[b]
        last_x, last, s_true = history[-1]
        if b < len(history) - 1 and last > 1.5 * best and last - best > 0.05 * s_true:
            results.append(_result(
                ctx, 'warning', "2D prediction error regressed",
                f"RMSE of '{pk}' vs '{tk}' was best at epoch {format_x(best_x)} ({fmt(best)}) and is now "
                f"{fmt(last)} ({last / best if best else float('inf'):.1f}×).",
                f"Keep the checkpoint from epoch {format_x(best_x)}; check for overfitting or an unstable "
                "learning rate.", records[-1], metric=name,
            ))
        elif len(history) >= 5 and history[0][1] > 0 and best >= 0.98 * history[0][1]:
            results.append(_result(
                ctx, 'info', "2D prediction error not improving",
                f"RMSE of '{pk}' vs '{tk}' has not improved on its first value ({fmt(history[0][1])}) "
                f"over {len(history)} records.",
                "If this head is supposed to be learning, check that its loss term is included and weighted.",
                records[-1], metric=name,
            ))
    return results


@diagnostic_rule(name="series_2d_collapse", description="A standalone 2D series collapsed to a single point")
def check_series_2d_collapse(ctx: RunContext):
    records = ctx.records(FILE_2D)
    if not records:
        return None
    latest = records[-1]
    keys = point_keys(latest)
    paired = {k for _, tk, pk in pair_keys(keys) for k in (tk, pk)}
    collapsed = []
    for k in keys:
        if k in paired:
            continue
        pts = valid_points(latest[k])
        if len(pts) >= 3 and spread(pts) == 0:
            collapsed.append(f"'{k}' at ({fmt(pts[0][0])}, {fmt(pts[0][1])})")
    if not collapsed:
        return None
    return _result(
        ctx, 'info', "2D series collapsed to a point",
        f"Every point is identical in {', '.join(collapsed)}.",
        "If these should vary, the producing code is likely returning a constant.", latest,
    )
