"""Scenario tests for xscope diagnostics.

Healthy runs must not raise warnings (false alarms teach users to ignore the panel);
each pathological run must trigger the rule written for it.
"""
import json
import math
import os
import random
import statistics
import time

import pytest

from xscope.data_manager import RunDataManager
from xscope.diagnostics import run_diagnostics_for_runs
from xscope.diagnostics.series import classify_metric


@pytest.fixture
def make_run(tmp_path):
    counter = {'n': 0}

    def _make(name, metrics=None, matrices=None, points=None, commit="abc1234", timestamp=None,
              config=None, status=None, calibration=None):
        counter['n'] += 1
        run_dir = tmp_path / f"exp{counter['n']:03d}_{name}"
        run_dir.mkdir()
        meta = {
            "experiment_name": name, "experiment_number": counter['n'],
            "timestamp": timestamp or time.strftime("%Y%m%d_%H%M%S"), "git_commit": commit,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta))
        for filename, rows in (("metrics.jsonl", metrics), ("matrix.jsonl", matrices), ("2d.jsonl", points),
                               ("calibration.jsonl", calibration)):
            if rows is not None:
                (run_dir / filename).write_text("".join(json.dumps(r) + "\n" for r in rows))
        for filename, data in (("config.json", config), ("status.json", status)):
            if data is not None:
                (run_dir / filename).write_text(json.dumps(data))
        return str(run_dir)
    return _make


def diagnose(tmp_path):
    dm = RunDataManager(str(tmp_path))
    return run_diagnostics_for_runs(dm.load_runs_metadata(), dm)


def rules(results, severity=None):
    return {r.rule for r in results if severity is None or r.severity == severity}


def alarms(results):
    return [r for r in results if r.severity in ('critical', 'warning')]


# ---------------------------------------------------------------------------
# Synthetic curves
# ---------------------------------------------------------------------------

def healthy_curves(epochs=60, seed=0, train_key="loss/train", val_key="loss/val", acc_key="accuracy"):
    rng = random.Random(seed)
    rows = []
    for e in range(1, epochs + 1):
        base = 2.3 * math.exp(-e / 8)
        rows.append({
            "epoch": e,
            train_key: base + 0.05 + rng.uniform(0, 0.004),
            val_key: base + 0.04 + rng.uniform(0, 0.004),    # below train: dropout-style, benign
            acc_key: min(0.97, 1 - math.exp(-e / 6)) + rng.uniform(-0.003, 0.003),
        })
    return rows


def overfit_curves(epochs=40, train_key="loss/train", val_key="loss/val", seed=0):
    rng = random.Random(seed)
    return [{"epoch": e, train_key: 1 / e + rng.uniform(0, 0.005),
             val_key: 0.3 + 0.001 * (e - 15) ** 2 + rng.uniform(0, 0.005)} for e in range(1, epochs + 1)]


def confusion(k, n_per_class=50, acc=0.9, seed=0):
    rng = random.Random(seed)
    m = [[0] * k for _ in range(k)]
    for i in range(k):
        for _ in range(n_per_class):
            j = i if rng.random() < acc else rng.randrange(k)
            m[i][j] += 1
    return m


# ---------------------------------------------------------------------------
# Metric-name interpretation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key, kind, split, base", [
    ("loss/train", "loss", "train", "loss"),
    ("train_loss", "loss", "train", "loss"),
    ("valLoss", "loss", "val", "loss"),
    ("val/acc", "score", "val", "acc"),
    ("test_f1", "score", "val", "f1"),
    ("lr", "lr", None, "lr"),
    ("learning_rate", "lr", None, "learning/rate"),
    ("grad_norm", "grad", None, "grad/norm"),
    ("accuracy", "score", None, "accuracy"),
    ("throughput", "other", None, "throughput"),
])
def test_classify_metric(key, kind, split, base):
    m = classify_metric(key)
    assert (m.kind, m.split, m.base) == (kind, split, base)


# ---------------------------------------------------------------------------
# Healthy runs: no false alarms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("keys", [
    ("loss/train", "loss/val", "accuracy"),
    ("train_loss", "val_loss", "val_acc"),
    ("trainLoss", "validLoss", "validAccuracy"),
])
def test_healthy_run_has_no_alarms(tmp_path, make_run, keys):
    make_run("healthy", metrics=healthy_curves(train_key=keys[0], val_key=keys[1], acc_key=keys[2]))
    results = diagnose(tmp_path)
    assert alarms(results) == []


def test_healthy_converged_run_reports_plateau_only(tmp_path, make_run):
    rows = [{"epoch": e, "loss/train": max(0.02, math.exp(-e / 5)), "loss/val": max(0.05, math.exp(-e / 5) + 0.03),
             "acc": min(0.99, 1 - math.exp(-e / 4))} for e in range(1, 51)]
    make_run("converged", metrics=rows)
    results = diagnose(tmp_path)
    assert alarms(results) == []
    assert "plateau" in rules(results, 'info')


def test_noisy_per_step_loss_is_not_flagged(tmp_path, make_run):
    rng = random.Random(1)
    rows = [{"epoch": s, "loss/train": 2.0 + 1.0 * math.exp(-s / 300) + rng.gauss(0, 0.08)} for s in range(1, 1500)]
    make_run("per_step", metrics=rows)
    assert alarms(diagnose(tmp_path)) == []


def test_healthy_many_class_matrix_has_no_alarms(tmp_path, make_run):
    labels = [f"c{i}" for i in range(100)]
    make_run("many_classes", matrices=[{"step": 1, "labels": labels, "matrix": confusion(100, acc=0.92)}])
    assert alarms(diagnose(tmp_path)) == []


def test_healthy_2d_predictions_have_no_alarms(tmp_path, make_run):
    rng = random.Random(0)
    truth = [[float(i), float(i) * 1.5] for i in range(20)]
    points = [{"epoch": e, "traj/gt": truth,
               "traj/pred": [[x + rng.gauss(0, 1 / e), y + rng.gauss(0, 1 / e)] for x, y in truth]}
              for e in range(1, 11)]
    make_run("good_2d", points=points)
    assert alarms(diagnose(tmp_path)) == []


# ---------------------------------------------------------------------------
# Scalar pathologies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("train_key, val_key", [("loss/train", "loss/val"), ("train_loss", "val_loss")])
def test_overfitting_detected_regardless_of_naming(tmp_path, make_run, train_key, val_key):
    make_run("overfit", metrics=overfit_curves(train_key=train_key, val_key=val_key))
    results = diagnose(tmp_path)
    over = [r for r in results if r.rule == "overfitting"]
    assert len(over) == 1 and over[0].severity == 'warning'
    assert 13 <= over[0].epoch <= 17          # best val epoch ≈ 15
    assert "plateau" not in rules(results)    # no contradicting "converged" note


def test_overfitting_notes_overconfidence_when_accuracy_holds(tmp_path, make_run):
    rows = overfit_curves()
    for r in rows:
        r["val_acc"] = min(0.9, 0.5 + 0.02 * r["epoch"])
    make_run("overconfident", metrics=rows)
    over = [r for r in diagnose(tmp_path) if r.rule == "overfitting"]
    assert over and "over-confident" in over[0].message


def test_divergence_is_critical(tmp_path, make_run):
    rows = [{"epoch": e, "loss": 1.0 * math.exp(-e / 5) if e < 15 else 0.05 * 1.6 ** (e - 15)} for e in range(1, 30)]
    make_run("diverged", metrics=rows)
    results = diagnose(tmp_path)
    assert "loss_divergence" in rules(results, 'critical')


def test_nan_is_critical(tmp_path, make_run):
    rows = healthy_curves(epochs=12)
    rows[-1]["loss/train"] = float('nan')
    make_run("nan", metrics=rows)
    nan = [r for r in diagnose(tmp_path) if r.rule == "nonfinite_values"]
    assert nan and nan[0].severity == 'critical' and nan[0].epoch == 12


def test_not_learning(tmp_path, make_run):
    rng = random.Random(0)
    rows = [{"epoch": e, "loss/train": 2.302 + rng.uniform(-0.005, 0.005), "lr": 1e-7} for e in range(1, 21)]
    make_run("stuck", metrics=rows)
    stuck = [r for r in diagnose(tmp_path) if r.rule == "not_learning"]
    assert stuck and "1e-07" in stuck[0].message


def test_recovered_spike_is_info(tmp_path, make_run):
    rows = [{"epoch": e, "loss": 1.0 * math.exp(-e / 20) + 0.1} for e in range(1, 61)]
    rows[30]["loss"] *= 4
    make_run("spike", metrics=rows)
    spikes = [r for r in diagnose(tmp_path) if r.rule == "loss_spikes"]
    assert spikes and spikes[0].severity == 'info' and spikes[0].epoch == 31


def test_spike_mentions_lr_increase(tmp_path, make_run):
    rows = [{"epoch": e, "loss": math.exp(-e / 20) + 0.1, "lr": 0.001 if e < 31 else 0.01} for e in range(1, 61)]
    for r in rows[30:]:
        r["loss"] *= 3
    make_run("lr_restart", metrics=rows)
    spikes = [r for r in diagnose(tmp_path) if r.rule == "loss_spikes"]
    assert spikes and "learning-rate increase" in spikes[0].message


def test_growing_noise_is_instability(tmp_path, make_run):
    rng = random.Random(0)
    rows = [{"epoch": e, "loss": 1.0 + rng.gauss(0, 0.01 if e < 40 else 0.15)} for e in range(1, 61)]
    rows = [{"epoch": r["epoch"], "loss": r["loss"] + 2 * math.exp(-r["epoch"] / 10)} for r in rows]
    make_run("unstable", metrics=rows)
    assert "training_instability" in rules(diagnose(tmp_path), 'warning')


def test_accuracy_regression(tmp_path, make_run):
    rows = [{"epoch": e, "accuracy": 0.5 + 0.04 * e if e <= 10 else 0.9 - 0.05 * (e - 10)} for e in range(1, 16)]
    make_run("acc_drop", metrics=rows)
    reg = [r for r in diagnose(tmp_path) if r.rule == "score_regression"]
    assert reg and reg[0].epoch == 10


def test_integrity_checks(tmp_path, make_run):
    rows = [{"epoch": e % 10 + 1, "loss/train": 1 / (e + 1), "accuracy": 50 + e * 10, "frozen_acc": 0.5,
             "mse": -0.1} for e in range(20)]
    make_run("broken", metrics=rows)
    titles = {r.title for r in diagnose(tmp_path) if r.rule == "metric_integrity"}
    assert titles == {"Epoch counter went backwards", "Metric never changes",
                      "Metric out of valid range", "Impossible negative loss"}


def test_vanished_metric(tmp_path, make_run):
    rows = [{"epoch": e, "loss/train": 1 / e, **({"loss/val": 1 / e} if e <= 10 else {})} for e in range(1, 31)]
    make_run("vanished", metrics=rows)
    assert any(r.title == "Metric stopped being logged" for r in diagnose(tmp_path))


def test_sparse_validation_is_not_vanished(tmp_path, make_run):
    rows = [{"epoch": e, "loss/train": 1 / e, **({"loss/val": 1 / e} if e % 10 == 0 else {})} for e in range(1, 31)]
    make_run("sparse_val", metrics=rows)
    assert not any(r.title == "Metric stopped being logged" for r in diagnose(tmp_path))


@pytest.mark.parametrize("values, title", [
    ([1.0] * 10 + [20.0, 30.0, 40.0], "Exploding gradients"),
    ([1.0] * 10 + [1e-5] * 5, "Vanishing gradients"),
])
def test_gradient_health(tmp_path, make_run, values, title):
    make_run("grads", metrics=[{"epoch": i + 1, "grad_norm": v} for i, v in enumerate(values)])
    assert title in {r.title for r in diagnose(tmp_path)}


def test_stopped_while_improving(tmp_path, make_run):
    started = time.time() - 3 * 3600
    path = make_run("killed", metrics=[{"epoch": e, "loss": 2.0 - 0.05 * e} for e in range(1, 31)],
                    timestamp=time.strftime("%Y%m%d_%H%M%S", time.localtime(started)))
    last_write = started + 600
    os.utime(os.path.join(path, "metrics.jsonl"), (last_write, last_write))
    assert "stopped_while_improving" in rules(diagnose(tmp_path))


def test_empty_run(tmp_path, make_run):
    make_run("empty")
    assert "empty_run" in rules(diagnose(tmp_path))


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def matrix_titles(tmp_path, make_run, matrix, labels=None, history=()):
    records = [{"step": i + 1, "labels": labels, "matrix": m} for i, m in enumerate([*history, matrix])]
    make_run("matrix", matrices=records)
    return {r.title for r in diagnose(tmp_path) if r.metric == 'matrix'}


def test_matrix_collapse(tmp_path, make_run):
    titles = matrix_titles(tmp_path, make_run, [[95, 2, 3], [88, 1, 1], [90, 0, 0]], ["cat", "dog", "bird"])
    assert "Predictions collapsed onto one class" in titles


def test_matrix_chance_level(tmp_path, make_run):
    m = [[10, 10, 10, 10] for _ in range(4)]
    assert "No better than a trivial baseline" in matrix_titles(tmp_path, make_run, m)


def test_matrix_hidden_minority_failure(tmp_path, make_run):
    m = [[900, 5, 5], [20, 25, 5], [20, 5, 25]]
    assert "Accuracy hides poor minority-class performance" in matrix_titles(tmp_path, make_run, m)


def test_matrix_weak_class_and_confusion_pair(tmp_path, make_run):
    m = confusion(5, acc=0.95)
    m[2] = [0, 2, 8, 40, 0]   # class 2 mostly predicted as class 3
    titles = matrix_titles(tmp_path, make_run, m)
    assert {"Classes with very low recall", "Systematic class confusion"} <= titles


def test_matrix_class_regression(tmp_path, make_run):
    before = [[45, 5, 0], [5, 45, 0], [0, 5, 45]]
    after = [[45, 5, 0], [30, 20, 0], [0, 5, 45]]
    assert "Class recall regressed" in matrix_titles(tmp_path, make_run, after, history=[before])


def test_matrix_malformed(tmp_path, make_run):
    assert "Malformed confusion matrix" in matrix_titles(tmp_path, make_run, [[1, 2, 3], [4, 5, 6]])


# ---------------------------------------------------------------------------
# 2D predictions
# ---------------------------------------------------------------------------

def titles_2d(tmp_path, make_run, records):
    make_run("pts", points=records)
    return {r.title for r in diagnose(tmp_path)}


TRUTH = [[float(i), float(i % 7)] for i in range(30)]


def test_2d_collapse(tmp_path, make_run):
    assert "2D predictions collapsed" in titles_2d(tmp_path, make_run, [
        {"epoch": 1, "data/true": TRUTH, "data/pred": [[14.5 + 0.01 * i, 3.0] for i in range(30)]}])


def test_2d_overdispersed(tmp_path, make_run):
    assert "2D predictions over-dispersed" in titles_2d(tmp_path, make_run, [
        {"epoch": 1, "data/true": TRUTH, "data/pred": [[x * 100, y * 100] for x, y in TRUTH]}])


def test_2d_offset(tmp_path, make_run):
    assert "Systematic 2D offset" in titles_2d(tmp_path, make_run, [
        {"epoch": 1, "data/true": TRUTH, "data/pred": [[x + 3, y - 2] for x, y in TRUTH]}])


def test_2d_outliers(tmp_path, make_run):
    pred = [[x + 0.1, y] for x, y in TRUTH]
    pred[4] = [500.0, 500.0]
    assert "2D outlier predictions" in titles_2d(tmp_path, make_run, [{"epoch": 1, "gt": TRUTH, "pred": pred}])


def test_2d_error_regression(tmp_path, make_run):
    records = [{"epoch": e, "data/true": TRUTH, "data/pred": [[x + 0.2 * abs(e - 3), y] for x, y in TRUTH]}
               for e in range(1, 8)]
    assert "2D prediction error regressed" in titles_2d(tmp_path, make_run, records)


def test_2d_nan(tmp_path, make_run):
    pred = [[x, y] for x, y in TRUTH]
    pred[0] = [float('nan'), 1.0]
    assert "NaN/Inf in 2D points" in titles_2d(tmp_path, make_run, [{"epoch": 1, "data/true": TRUTH, "data/pred": pred}])


# ---------------------------------------------------------------------------
# Cross-run
# ---------------------------------------------------------------------------

def test_outlier_run(tmp_path, make_run):
    for seed in range(3):
        make_run(f"ok{seed}", metrics=healthy_curves(seed=seed))
    bad = healthy_curves(seed=9)
    for r in bad:
        r["loss/val"] += 2.0
    make_run("bad", metrics=bad)
    outliers = [r for r in diagnose(tmp_path) if r.rule == "outlier_runs"]
    assert [r.run_name for r in outliers] == ["bad #4"]


def test_metric_name_mismatch(tmp_path, make_run):
    make_run("a", metrics=healthy_curves())
    make_run("b", metrics=healthy_curves(train_key="train_loss", val_key="val_loss"))
    assert "metric_name_mismatch" in rules(diagnose(tmp_path))


def test_rule_crash_is_isolated(tmp_path, make_run, monkeypatch):
    from xscope.diagnostics import DIAGNOSTIC_RULES

    def boom(ctx):
        raise RuntimeError("bug in a rule")
    boom._diag_meta = {"name": "boom", "description": "", "enabled": True}
    monkeypatch.setitem(DIAGNOSTIC_RULES, "boom", boom)
    make_run("nan", metrics=[{"epoch": 1, "loss": float('nan')}])
    assert "nonfinite_values" in rules(diagnose(tmp_path))


def test_false_alarm_rate_on_random_healthy_runs(tmp_path, make_run):
    """Random healthy curves (shapes, lengths, noise, val offsets) should almost never raise alarms."""
    rng = random.Random(2024)
    n_runs = 300
    for k in range(n_runs):
        n = rng.choice([10, 30, 60, 120, 400])
        noise = rng.choice([0.002, 0.01, 0.03, 0.06])
        l0, lmin = rng.uniform(0.5, 5), rng.uniform(0.01, 0.4)
        tau, val_off, power = n * rng.uniform(0.1, 0.5), rng.uniform(-0.05, 0.15), rng.random() < 0.5
        rows = []
        for e in range(1, n + 1):
            base = lmin + (l0 - lmin) * (1 / math.sqrt(1 + 5 * e / tau) if power else math.exp(-e / tau))
            rows.append({"epoch": e, "loss/train": base * (1 + rng.gauss(0, noise)),
                         "loss/val": base * (1 + val_off) * (1 + rng.gauss(0, 1.5 * noise))})
        make_run(f"h{k}", metrics=rows)
    flagged = {r.run_name for r in alarms(diagnose(tmp_path)) if r.rule != "outlier_runs"}
    assert len(flagged) / n_runs < 0.01, flagged



# ---------------------------------------------------------------------------
# Generalisation gap and learning-rate notes
# ---------------------------------------------------------------------------

def test_score_gap_with_memorization(tmp_path, make_run):
    rows = [{"epoch": e, "train_acc": min(1.0, 0.5 + 0.05 * e), "val_acc": min(0.7, 0.45 + 0.03 * e)}
            for e in range(1, 31)]
    make_run("memorize", metrics=rows)
    gap = [r for r in diagnose(tmp_path) if r.rule == "score_gap"]
    assert gap and gap[0].severity == 'warning' and "memorized" in gap[0].message


def test_small_score_gap_is_quiet(tmp_path, make_run):
    rows = [{"epoch": e, "train_acc": min(0.95, 0.5 + 0.05 * e), "val_acc": min(0.91, 0.48 + 0.05 * e)}
            for e in range(1, 31)]
    make_run("fine", metrics=rows)
    assert "score_gap" not in rules(diagnose(tmp_path))


def test_early_spike_without_warmup_mentions_warmup(tmp_path, make_run):
    rows = [{"epoch": e, "loss": math.exp(-e / 20) + 0.1, "lr": 0.1} for e in range(1, 81)]
    rows[6]["loss"] *= 4
    make_run("no_warmup", metrics=rows)
    spikes = [r for r in diagnose(tmp_path) if r.rule == "loss_spikes"]
    assert spikes and "warmup" in spikes[0].message


def test_plateau_mentions_missing_lr_decay(tmp_path, make_run):
    rows = [{"epoch": e, "loss": max(0.3, math.exp(-e / 5)), "lr": 0.01} for e in range(1, 41)]
    make_run("flat_lr", metrics=rows)
    plateau = [r for r in diagnose(tmp_path) if r.rule == "plateau"]
    assert plateau and "never decayed" in plateau[0].message


def test_cosine_schedule_is_not_called_undertrained(tmp_path, make_run):
    loss, rows = 2.5, []
    for e in range(1, 61):
        lr = 0.5 * (1 + math.cos(math.pi * e / 60))
        loss -= lr * 0.08 * (loss - 0.2)
        rows.append({"epoch": e, "loss/train": loss, "lr": 0.1 * lr})
    make_run("cosine", metrics=rows, status={"status": "finished"})
    assert "stopped_while_improving" not in rules(diagnose(tmp_path))


def test_finished_while_improving(tmp_path, make_run):
    make_run("short", metrics=[{"epoch": e, "loss": 2.0 - 0.05 * e} for e in range(1, 31)],
             status={"status": "finished"})
    assert "stopped_while_improving" in rules(diagnose(tmp_path))


# ---------------------------------------------------------------------------
# Run lifecycle (status.json and per-record _time from current mtrick)
# ---------------------------------------------------------------------------

def timed_rows(n, start, gaps):
    t, rows = start, []
    for e in range(1, n + 1):
        t += gaps(e)
        rows.append({"epoch": e, "loss": 1 / e, "_time": t})
    return rows


def test_crashed_run_reports_error(tmp_path, make_run):
    make_run("oom", metrics=timed_rows(5, time.time() - 100, lambda e: 1),
             status={"status": "crashed", "error": "RuntimeError: CUDA out of memory"})
    crash = [r for r in diagnose(tmp_path) if r.rule == "run_status"]
    assert crash and crash[0].severity == 'critical'
    assert "epoch 5" in crash[0].message and "batch size" in crash[0].suggested_action


def test_interrupted_run(tmp_path, make_run):
    make_run("ctrl_c", metrics=timed_rows(5, time.time() - 100, lambda e: 1), status={"status": "interrupted"})
    assert "Run interrupted" in {r.title for r in diagnose(tmp_path)}


def test_unresponsive_running_run(tmp_path, make_run):
    start = time.time() - 4 * 3600
    make_run("killed", metrics=timed_rows(20, start, lambda e: 30),
             status={"status": "running", "started": start, "updated": start})
    assert "Run stopped responding" in {r.title for r in diagnose(tmp_path)}


def test_live_running_run_is_quiet(tmp_path, make_run):
    start = time.time() - 600
    make_run("live", metrics=timed_rows(20, start, lambda e: 29),
             status={"status": "running", "started": start, "updated": start})
    assert "run_status" not in rules(diagnose(tmp_path))


def test_training_slowdown(tmp_path, make_run):
    make_run("slow", metrics=timed_rows(30, 0, lambda e: 10 if e < 15 else 35))
    assert "training_slowdown" in rules(diagnose(tmp_path), 'warning')


def test_steady_speed_is_quiet(tmp_path, make_run):
    rng = random.Random(0)
    make_run("steady", metrics=timed_rows(30, 0, lambda e: 10 + rng.uniform(-2, 2)))
    assert "training_slowdown" not in rules(diagnose(tmp_path))


def test_memory_creep(tmp_path, make_run):
    rows = [{"epoch": e, "loss": 1 / e, "sys/peak_rss_gb": 2.0 + 0.1 * e} for e in range(1, 31)]
    make_run("leak", metrics=rows)
    assert "memory_creep" in rules(diagnose(tmp_path), 'warning')


def test_memory_that_plateaus_after_warmup_is_quiet(tmp_path, make_run):
    rows = [{"epoch": e, "loss": 1 / e, "sys/gpu_mem_gb": min(8.0, 2.0 + 2 * e)} for e in range(1, 31)]
    make_run("ok_mem", metrics=rows)
    assert "memory_creep" not in rules(diagnose(tmp_path))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def reliability(pairs, n=100):
    return [{"count": n, "confidence": c, "accuracy": a} for c, a in pairs]


def test_overconfident_model(tmp_path, make_run):
    make_run("overconf", calibration=[
        {"step": 1, "bins": reliability([(0.55, 0.53), (0.75, 0.74), (0.95, 0.93)])},
        {"step": 2, "bins": reliability([(0.55, 0.40), (0.75, 0.55), (0.95, 0.75)])},
    ])
    cal = [r for r in diagnose(tmp_path) if r.rule == "calibration"]
    assert cal and cal[0].title == "Over-confident predictions" and cal[0].severity == 'warning'
    assert "worsened" in cal[0].message


def test_calibrated_model_is_quiet(tmp_path, make_run):
    make_run("calibrated", calibration=[{"step": 1, "bins": reliability([(0.55, 0.56), (0.75, 0.74), (0.95, 0.94)])}])
    assert "calibration" not in rules(diagnose(tmp_path))


# ---------------------------------------------------------------------------
# Confusion matrix additions
# ---------------------------------------------------------------------------

def test_matrix_precision_regression(tmp_path, make_run):
    before = [[45, 5, 0], [5, 45, 0], [0, 5, 45]]
    after = [[25, 25, 0], [5, 45, 0], [0, 30, 20]]   # many inputs now predicted as class 1
    assert "Class precision regressed" in matrix_titles(tmp_path, make_run, after, history=[before])


def test_logged_accuracy_mismatch(tmp_path, make_run):
    m = [[30, 20], [20, 30]]  # 60% accuracy
    make_run("mismatch", metrics=[{"epoch": 5, "val_acc": 0.95}], matrices=[{"step": 5, "matrix": m}])
    assert "matrix_scalar_mismatch" in rules(diagnose(tmp_path), 'warning')


def test_logged_accuracy_matches_in_percent(tmp_path, make_run):
    m = [[30, 20], [20, 30]]
    make_run("match", metrics=[{"epoch": 5, "val_acc": 60.0}], matrices=[{"step": 5, "matrix": m}])
    assert "matrix_scalar_mismatch" not in rules(diagnose(tmp_path))


# ---------------------------------------------------------------------------
# 2D error structure
# ---------------------------------------------------------------------------

LINE = [[float(i), float(i) * 0.5 + (i % 3)] for i in range(24)]


def test_2d_drift_along_sequence(tmp_path, make_run):
    pred = [[x + 0.05 * i * i / 4, y] for i, (x, y) in enumerate(LINE)]
    assert "2D error accumulates along the sequence" in titles_2d(
        tmp_path, make_run, [{"epoch": 1, "traj/gt": LINE, "traj/pred": pred}])


def test_2d_error_on_one_axis(tmp_path, make_run):
    rng = random.Random(0)
    pred = [[x + rng.gauss(0, 3), y + rng.gauss(0, 0.05)] for x, y in LINE]
    assert "2D error concentrated on one axis" in titles_2d(
        tmp_path, make_run, [{"epoch": 1, "data/true": LINE, "data/pred": pred}])


def test_2d_scale_bias(tmp_path, make_run):
    cx, cy = statistics.fmean(p[0] for p in LINE), statistics.fmean(p[1] for p in LINE)
    pred = [[cx + 0.6 * (x - cx), cy + 0.6 * (y - cy)] for x, y in LINE]
    assert "2D predictions scaled relative to the truth" in titles_2d(
        tmp_path, make_run, [{"epoch": 1, "data/true": LINE, "data/pred": pred}])


# ---------------------------------------------------------------------------
# Cross-run additions
# ---------------------------------------------------------------------------

def acc_rows(final, noise, seed):
    rng = random.Random(seed)
    return [{"epoch": e, "val_acc": final * (1 - math.exp(-e / 5)) + rng.gauss(0, noise)} for e in range(1, 41)]


def test_runs_within_noise(tmp_path, make_run):
    make_run("a", metrics=acc_rows(0.900, 0.01, 1))
    make_run("b", metrics=acc_rows(0.897, 0.01, 2))
    assert "inconclusive_comparison" in rules(diagnose(tmp_path))


def test_clearly_different_runs_are_conclusive(tmp_path, make_run):
    make_run("a", metrics=acc_rows(0.90, 0.002, 1))
    make_run("b", metrics=acc_rows(0.80, 0.002, 2))
    assert "inconclusive_comparison" not in rules(diagnose(tmp_path))


def test_config_attribution(tmp_path, make_run):
    diverging = [{"epoch": e, "loss": math.exp(-e / 5) if e < 15 else 0.05 * 1.6 ** (e - 15)} for e in range(1, 30)]
    for i in range(2):
        make_run(f"hot{i}", metrics=diverging, config={"lr": 0.1 * (i + 1), "batch_size": 64, "seed": i})
        make_run(f"ok{i}", metrics=healthy_curves(seed=i), config={"lr": 0.001, "batch_size": 64, "seed": 10 + i})
    found = [r for r in diagnose(tmp_path) if r.rule == "config_attribution"]
    assert found and "lr ≥ 0.1" in found[0].message
    assert "batch_size" not in found[0].message and "seed" not in found[0].message


def test_config_type_mismatch(tmp_path, make_run):
    make_run("a", metrics=healthy_curves(), config={"lr": 0.001})
    make_run("b", metrics=healthy_curves(seed=1), config={"lr": "1e-3"})
    assert "config_consistency" in rules(diagnose(tmp_path))


def test_outlier_run_shows_config_difference(tmp_path, make_run):
    for seed in range(3):
        make_run(f"ok{seed}", metrics=healthy_curves(seed=seed), config={"dropout": 0.1, "lr": 0.001})
    bad = healthy_curves(seed=9)
    for r in bad:
        r["loss/val"] += 2.0
    make_run("bad", metrics=bad, config={"dropout": 0.0, "lr": 0.001})
    outlier = [r for r in diagnose(tmp_path) if r.rule == "outlier_runs"]
    assert outlier and "dropout=0.0 (others 0.1)" in outlier[0].message
