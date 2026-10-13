"""Diagnostics about the training process itself: run status, speed and memory.

Needs status.json and per-record '_time' (written by current mtrick); runs from older mtrick versions are skipped.
"""
import statistics
import time

from xscope.diagnostics.core import RunContext, diagnostic_rule
from xscope.diagnostics.rules_scalar import METRIC_FILES, _duration, _result
from xscope.diagnostics.series import fmt, format_x


def _last_epoch(ctx: RunContext) -> str:
    records = ctx.records()
    if not records:
        return "the first log() call"
    r = records[-1]
    return f"epoch {format_x(r.get('epoch', r.get('step', len(records))))}"


def typical_interval(ctx: RunContext):
    """Median seconds between consecutive metric records, or None without timestamps."""
    times = ctx.times()
    gaps = [b - a for a, b in zip(times, times[1:]) if b >= a]
    return statistics.median(gaps) if len(gaps) >= 3 else None


@diagnostic_rule(name="run_status", description="Crashed, interrupted or unresponsive runs")
def check_run_status(ctx: RunContext):
    status = ctx.status()
    if not status:
        return None
    state = status.get('status')

    if state == 'crashed':
        error = status.get('error') or "an exception"
        oom = 'out of memory' in error.lower() or 'memoryerror' in error.lower()
        return _result(
            ctx, 'critical', "Run crashed",
            f"The training process raised {error} after {_last_epoch(ctx)}.",
            "Reduce the batch size, use gradient accumulation or mixed precision, and free tensors you keep "
            "around between steps." if oom else "See the full traceback in the process output or job log.",
        )
    if state == 'interrupted':
        return _result(
            ctx, 'info', "Run interrupted",
            f"The run was stopped manually (KeyboardInterrupt) after {_last_epoch(ctx)}.",
            None,
        )
    if state == 'running':
        last = max([status.get('updated') or 0, status.get('started') or 0]
                   + [t for f in METRIC_FILES for t in ctx.times(f)])
        if not last:
            return None
        idle = time.time() - last
        interval = typical_interval(ctx)
        limit = max(10 * interval, 600) if interval else 3600
        if idle > limit:
            cadence = f" (it normally logs every ~{_duration(interval)})" if interval else ""
            return _result(
                ctx, 'warning', "Run stopped responding",
                f"Status is still 'running', but nothing has been logged for {_duration(idle)}{cadence}. "
                "The process was most likely killed (out-of-memory killer, preemption, SIGKILL) or is hung.",
                "Check the job or system logs (e.g. `dmesg` for the OOM killer). In notebooks, call "
                "tracker.finish() or use `with Tracker(...)` so finished runs are marked as such.",
            )
    return None


@diagnostic_rule(name="training_slowdown", description="Epochs taking much longer than earlier in the run")
def check_training_slowdown(ctx: RunContext):
    times = ctx.times()
    gaps = [b - a for a, b in zip(times, times[1:]) if b >= a]
    if len(gaps) < 12:
        return None
    third = len(gaps) // 3
    early, late = statistics.median(gaps[:third]), statistics.median(gaps[-third:])
    if early <= 0 or late <= 2 * early or late - early < 1.0:
        return None
    return _result(
        ctx, 'warning', "Training slowing down",
        f"Time between logged records grew {late / early:.1f}× during the run (~{_duration(early)} → "
        f"~{_duration(late)}).",
        "Common causes: a memory leak causing swapping or GC pressure (check memory metrics), tensors kept "
        "alive across steps, a growing cache or dataset, or thermal throttling.",
    )


@diagnostic_rule(name="memory_creep", description="Memory usage that keeps growing (likely a leak)")
def check_memory_creep(ctx: RunContext):
    results = []
    for m in ctx.metrics_of_kind('memory'):
        _, ys = ctx.series(m.key)
        if len(ys) < 12:
            continue
        third = len(ys) // 3
        a = statistics.median(ys[:third])
        b = statistics.median(ys[third:-third])
        c = statistics.median(ys[-third:])
        if a > 0 and c > 1.3 * a and b > 1.05 * a and c > 1.05 * b:
            results.append(_result(
                ctx, 'warning', "Memory keeps growing",
                f"'{m.key}' rose from {fmt(a)} to {fmt(c)} and is still increasing in the last third of training.",
                "Look for tensors accumulated across steps (e.g. appending `loss` instead of `loss.item()`), "
                "growing caches, or DataLoader workers leaking memory.",
                metric=m.key,
            ))
    return results
