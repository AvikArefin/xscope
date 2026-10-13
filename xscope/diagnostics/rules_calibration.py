"""Calibration diagnostics from reliability histograms in calibration.jsonl (mtrick log_calibration)."""
from typing import Optional

from xscope.diagnostics.core import DiagnosticResult, RunContext, diagnostic_rule
from xscope.diagnostics.series import format_x, is_finite_number

CALIBRATION_FILE = "calibration.jsonl"
MIN_SAMPLES = 50


def calibration_summary(record: dict) -> Optional[dict]:
    """ECE and mean confidence/accuracy of one reliability histogram, or None if unusable."""
    bins = record.get('bins')
    if not isinstance(bins, list):
        return None
    usable = [b for b in bins if isinstance(b, dict) and all(is_finite_number(b.get(k))
                                                            for k in ('count', 'confidence', 'accuracy'))]
    total = sum(b['count'] for b in usable)
    if total < MIN_SAMPLES:
        return None
    return {
        'ece': sum(b['count'] / total * abs(b['accuracy'] - b['confidence']) for b in usable),
        'confidence': sum(b['count'] / total * b['confidence'] for b in usable),
        'accuracy': sum(b['count'] / total * b['accuracy'] for b in usable),
        'step': record.get('step', record.get('epoch')),
        'n': total,
    }


@diagnostic_rule(name="calibration", description="Over- or under-confident predictions")
def check_calibration(ctx: RunContext):
    history = [s for s in (calibration_summary(r) for r in ctx.records(CALIBRATION_FILE)) if s]
    if not history:
        return None
    latest = history[-1]
    ece = latest['ece']
    if ece < 0.05:
        return None
    over = latest['confidence'] > latest['accuracy']
    trend = ""
    first = history[0]
    if len(history) >= 2 and first['ece'] < 0.5 * ece and ece - first['ece'] > 0.03:
        trend = f" It has worsened from {first['ece']:.1%} at step {format_x(first['step'])}."
    step = latest['step']
    return DiagnosticResult(
        'warning' if ece >= 0.1 else 'info',
        "Over-confident predictions" if over else "Under-confident predictions",
        f"Expected calibration error is {ece:.1%}: average confidence {latest['confidence']:.1%} vs accuracy "
        f"{latest['accuracy']:.1%} over {latest['n']:g} predictions.{trend}",
        ctx.name,
        epoch=int(step) if is_finite_number(step) and float(step).is_integer() else None,
        suggested_action=(
            "Fit temperature scaling on a validation set, or use label smoothing / mixup during training; "
            "over-confidence usually grows as a model overfits." if over else
            "Temperature scaling (temperature < 1) corrects systematic under-confidence; heavy label smoothing "
            "is a common cause."
        ),
        metric='calibration',
    )
