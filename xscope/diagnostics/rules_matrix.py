"""Diagnostics for confusion matrices in matrix.jsonl (rows = true class, columns = predicted)."""
from dataclasses import dataclass
from typing import Optional

from xscope.diagnostics.core import DiagnosticResult, RunContext, diagnostic_rule
from xscope.diagnostics.series import format_x, is_finite_number

MATRIX_FILE = "matrix.jsonl"
MIN_SAMPLES = 20        # below this, per-class statistics are too noisy to judge
MIN_CLASS_SAMPLES = 5   # per-class rates need at least this many true samples


@dataclass
class Confusion:
    labels: list[str]
    m: list[list[float]]
    step: object

    def __post_init__(self):
        k = len(self.m)
        self.k = k
        self.rows = [sum(r) for r in self.m]
        self.cols = [sum(self.m[i][j] for i in range(k)) for j in range(k)]
        self.total = sum(self.rows)
        diag = [self.m[i][i] for i in range(k)]
        self.accuracy = sum(diag) / self.total if self.total else 0.0
        self.recall = [diag[i] / self.rows[i] if self.rows[i] else None for i in range(k)]
        self.precision = [diag[j] / self.cols[j] if self.cols[j] else None for j in range(k)]
        present = [r for r in self.recall if r is not None]
        self.balanced_accuracy = sum(present) / len(present) if present else 0.0
        self.present = [i for i in range(k) if self.rows[i] > 0]

    def name(self, i: int) -> str:
        return f"'{self.labels[i]}'"


def parse_confusion(record: dict) -> tuple[Optional[Confusion], Optional[str]]:
    """Returns (confusion, problem); problem describes why the record can't be analysed."""
    m = record.get('matrix')
    if not isinstance(m, list) or not m or not all(isinstance(r, list) for r in m):
        return None, "the 'matrix' field is missing or not a list of rows"
    k = len(m)
    if any(len(r) != len(m[0]) for r in m):
        return None, "rows have different lengths"
    if len(m[0]) != k:
        return None, f"it is {k}×{len(m[0])}, but a confusion matrix must be square"
    if not all(is_finite_number(v) for r in m for v in r):
        return None, "it contains non-numeric, NaN or Inf entries"
    if any(v < 0 for r in m for v in r):
        return None, "it contains negative counts"
    labels = record.get('labels')
    if not isinstance(labels, list) or len(labels) != k:
        labels = [f"Class {i}" for i in range(k)]
    return Confusion([str(l) for l in labels], m, record.get('step', record.get('epoch'))), None


def _latest(ctx: RunContext) -> tuple[Optional[Confusion], Optional[str], Optional[dict]]:
    records = ctx.records(MATRIX_FILE)
    if not records:
        return None, None, None
    c, problem = parse_confusion(records[-1])
    return c, problem, records[-1]


def _join(items: list[str], limit: int = 3) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def _pct(v: float) -> str:
    return f"{100 * v:.0f}%"


def _result(ctx: RunContext, severity: str, title: str, message: str, action: str | None,
            c: Optional[Confusion] = None) -> DiagnosticResult:
    step = c.step if c is not None else None
    epoch = int(step) if is_finite_number(step) and float(step).is_integer() else None
    return DiagnosticResult(severity, title, message, ctx.name, epoch=epoch,
                            suggested_action=action, metric='matrix')


@diagnostic_rule(name="matrix_integrity", description="Malformed matrices, missing classes, changing eval-set size")
def check_matrix_integrity(ctx: RunContext):
    records = ctx.records(MATRIX_FILE)
    if not records:
        return None
    results = []
    c, problem, latest = _latest(ctx)
    if problem:
        results.append(_result(
            ctx, 'warning', "Malformed confusion matrix",
            f"The latest matrix can't be analysed: {problem}.",
            "Log a square [true][predicted] count matrix, e.g. sklearn.metrics.confusion_matrix(y_true, y_pred).",
        ))
        return results

    labels = latest.get('labels')
    if isinstance(labels, list) and len(labels) != c.k:
        results.append(_result(
            ctx, 'warning', "Matrix labels don't match its size",
            f"{len(labels)} labels were logged for a {c.k}×{c.k} matrix, so classes may be mislabelled.",
            "Pass exactly one label per row/column, in the same order as the matrix.", c,
        ))

    missing = [c.name(i) for i in range(c.k) if c.rows[i] == 0]
    if missing and len(missing) < c.k:
        results.append(_result(
            ctx, 'info', "Classes missing from evaluation data",
            f"{len(missing)} of {c.k} classes have no true samples in the latest matrix: {_join(missing)}.",
            "Their recall can't be measured; make sure the evaluation split covers every class.", c,
        ))

    totals = [p.total for p in (parse_confusion(r)[0] for r in records) if p is not None and p.total > 0]
    if len(totals) >= 2 and max(totals) > 1.01 * min(totals):
        results.append(_result(
            ctx, 'info', "Evaluation set size changed",
            f"Samples per matrix ranged from {min(totals):g} to {max(totals):g} across steps, so steps are "
            "not compared on the same data.",
            "Evaluate on a fixed validation set (no random sub-sampling) to make steps comparable.", c,
        ))
    return results


@diagnostic_rule(name="matrix_quality", description="Collapse, chance-level, hidden minority failure, confusions")
def check_matrix_quality(ctx: RunContext):
    c, problem, _ = _latest(ctx)
    if c is None or c.total < MIN_SAMPLES or len(c.present) < 2:
        return None
    results = []
    share_true = [r / c.total for r in c.rows]
    share_pred = [col / c.total for col in c.cols]

    # 1. Prediction collapse: one class soaks up far more predictions than it has samples.
    j = max(range(c.k), key=lambda i: share_pred[i] - share_true[i])
    if (share_pred[j] >= max(0.5, share_true[j] + 0.3)
            or (share_pred[j] >= 0.85 and share_pred[j] >= share_true[j] + 0.15)):
        title = "Predictions collapsed onto one class" if share_pred[j] >= 0.85 else "Predictions biased toward one class"
        results.append(_result(
            ctx, 'warning', title,
            f"{_pct(share_pred[j])} of predictions are {c.name(j)}, which is only {_pct(share_true[j])} of the "
            f"true labels. Balanced accuracy is {_pct(c.balanced_accuracy)}.",
            "Check class balance and loss weighting, the learning rate (collapse is common when it is too high), "
            "and that labels are shuffled together with inputs.", c,
        ))
        return results

    # 2. No better than always predicting the most common class.
    majority = max(c.present, key=lambda i: c.rows[i])
    if c.accuracy <= share_true[majority] + 0.02:
        balanced = max(share_true) < 1.5 / len(c.present)
        baseline = f"chance level (1/{len(c.present)} = {_pct(1 / len(c.present))})" if balanced else \
            f"always predicting {c.name(majority)} ({_pct(share_true[majority])})"
        results.append(_result(
            ctx, 'warning', "No better than a trivial baseline",
            f"Accuracy is {_pct(c.accuracy)}, no better than {baseline}.",
            "The model hasn't learned to separate the classes yet; see the 'Not learning' checks for the loss.", c,
        ))
        return results

    ratio = max(c.rows[i] for i in c.present) / min(c.rows[i] for i in c.present)
    weak = sorted(
        (i for i in c.present
         if c.rows[i] >= MIN_CLASS_SAMPLES
         and (c.recall[i] <= 0.1 or (c.recall[i] < 0.5 and c.recall[i] < 0.5 * c.balanced_accuracy))),
        key=lambda i: c.recall[i],
    )
    weak_text = [f"{c.name(i)} {_pct(c.recall[i])}" for i in weak]

    # 3. Overall accuracy hiding failure on minority classes.
    if ratio >= 3 and c.accuracy - c.balanced_accuracy >= 0.1:
        detail = f" Weakest recall: {_join(weak_text)}." if weak_text else ""
        results.append(_result(
            ctx, 'warning', "Accuracy hides poor minority-class performance",
            f"Accuracy is {_pct(c.accuracy)} but balanced accuracy (mean per-class recall) is only "
            f"{_pct(c.balanced_accuracy)}; the largest class is {ratio:.0f}× the smallest.{detail}",
            "Track balanced accuracy or macro-F1, and use class weights, focal loss or resampling.", c,
        ))
    else:
        # 4. Individual classes the model mostly misses.
        if weak:
            results.append(_result(
                ctx, 'warning', "Classes with very low recall",
                f"Recall: {_join(weak_text)} (balanced accuracy {_pct(c.balanced_accuracy)}).",
                "Collect more examples of these classes, weight them higher in the loss, "
                "or inspect their samples for label noise.", c,
            ))
        # 5. Imbalanced evaluation data.
        if ratio >= 10:
            big = max(c.present, key=lambda i: c.rows[i])
            small = min(c.present, key=lambda i: c.rows[i])
            results.append(_result(
                ctx, 'info', "Imbalanced evaluation set",
                f"{c.name(big)} has {ratio:.0f}× more samples than {c.name(small)} "
                f"({c.rows[big]:g} vs {c.rows[small]:g}).",
                "Report balanced accuracy or macro-F1 alongside accuracy.", c,
            ))

    # 6. Classes predicted far more often than they occur (low precision sinks).
    over = sorted(
        (j for j in range(c.k)
         if c.cols[j] >= 1.5 * max(c.rows[j], 1) and share_pred[j] >= 0.05
         and c.precision[j] is not None and c.precision[j] < 0.5),
        key=lambda j: c.precision[j],
    )
    if over:
        items = [f"{c.name(j)} ({c.cols[j] / max(c.rows[j], 1):.1f}× its frequency, precision {_pct(c.precision[j])})"
                 for j in over]
        results.append(_result(
            ctx, 'info', "Over-predicted classes",
            f"Predicted far more often than they occur: {_join(items)}.",
            "The model uses these as a fallback for uncertain inputs; check their training share and decision threshold.", c,
        ))

    # 7. Systematic confusions between specific class pairs.
    min_count = max(3, 0.01 * c.total)
    pairs = []
    for i in c.present:
        for j in range(c.k):
            if i != j and c.m[i][j] >= min_count and c.m[i][j] / c.rows[i] >= 0.3:
                pairs.append((c.m[i][j] / c.rows[i], i, j))
    if pairs:
        pairs.sort(reverse=True)
        shown, items = set(), []
        for rate, i, j in pairs:
            if (j, i) in shown:
                continue
            shown.add((i, j))
            back = c.m[j][i] / c.rows[j] if c.rows[j] else 0
            if back >= 0.2:
                items.append(f"{c.name(i)} ↔ {c.name(j)} ({_pct(rate)} / {_pct(back)})")
            else:
                items.append(f"{c.name(i)} → {c.name(j)} ({_pct(rate)})")
        results.append(_result(
            ctx, 'info', "Systematic class confusion",
            f"Share of each true class predicted as another: {_join(items)}.",
            "Inspect the misclassified samples; ↔ pairs may need more distinguishing features, "
            "merged labels, or a cleaner labeling guideline.", c,
        ))
    return results


def _regressions(history: list[Confusion], rate: str, counts: str) -> list[tuple[float, str]]:
    """Classes whose per-class rate ('recall' or 'precision') fell well below its best earlier step."""
    latest = history[-1]
    found = []
    for i in range(latest.k):
        if getattr(latest, counts)[i] < MIN_CLASS_SAMPLES:
            continue
        candidates = [(getattr(c, rate)[i], c.step) for c in history[:-1]
                      if getattr(c, counts)[i] >= MIN_CLASS_SAMPLES and getattr(c, rate)[i] is not None]
        if not candidates:
            continue
        best, best_step = max(candidates, key=lambda t: t[0])
        now = getattr(latest, rate)[i]
        if best - now >= 0.2 and now < 0.8 * best:
            found.append((best - now, f"{latest.name(i)} {_pct(best)} (step {format_x(best_step)}) → {_pct(now)}"))
    return sorted(found, reverse=True)


@diagnostic_rule(name="matrix_class_regression", description="Per-class recall or precision dropping between steps")
def check_matrix_class_regression(ctx: RunContext):
    history = [c for c in (parse_confusion(r)[0] for r in ctx.records(MATRIX_FILE)) if c is not None]
    if len(history) < 2:
        return None
    latest = history[-1]
    history = [c for c in history if c.k == latest.k and c.labels == latest.labels]
    if len(history) < 2 or latest.total < MIN_SAMPLES:
        return None

    results = []
    recall = _regressions(history, 'recall', 'rows')
    if recall:
        results.append(_result(
            ctx, 'warning', "Class recall regressed",
            f"Recall fell for: {_join([text for _, text in recall])}.",
            "The model is trading these classes for others (or forgetting them); check class balance across "
            "batches and consider keeping the checkpoint where they peaked.", latest,
        ))
    precision = _regressions(history, 'precision', 'cols')
    if precision:
        results.append(_result(
            ctx, 'warning', "Class precision regressed",
            f"Precision fell for: {_join([text for _, text in precision])} — more inputs from other classes "
            "are now predicted as these.",
            "Inspect what is being misclassified into these classes; a shift in the training class mix "
            "or decision threshold is a common cause.", latest,
        ))
    return results


ACCURACY_BASES = {'acc', 'accuracy', 'top1', 'acc1'}


@diagnostic_rule(name="matrix_scalar_mismatch", description="Logged accuracy disagrees with the confusion matrix")
def check_matrix_scalar_mismatch(ctx: RunContext):
    matrices = [c for c in (parse_confusion(r)[0] for r in ctx.records(MATRIX_FILE))
                if c is not None and c.total > 0 and is_finite_number(c.step)]
    if not matrices:
        return None
    candidates = [m for m in ctx.metrics_of_kind('score')
                  if m.split != 'train' and set(m.base.split('/')) <= ACCURACY_BASES]
    for m in sorted(candidates, key=lambda m: m.split != 'val'):
        logged = dict(zip(*ctx.series(m.key)))
        worst = None
        for c in matrices:
            if c.step in logged:
                value = logged[c.step]
                fraction = value / 100 if value > 1 else value
                diff = abs(fraction - c.accuracy)
                if worst is None or diff > worst[0]:
                    worst = (diff, c, value)
        if worst and worst[0] > 0.02:
            _, c, value = worst
            return _result(
                ctx, 'warning', "Logged accuracy doesn't match the confusion matrix",
                f"At step {format_x(c.step)}, '{m.key}' is {value:g} but the confusion matrix gives "
                f"{_pct(c.accuracy)} ({c.total:g} samples).",
                "Both should come from the same predictions: check for different evaluation subsets, averaging "
                "per-batch accuracies instead of totals, or a matrix computed at a different step.", c,
            )
    return None
