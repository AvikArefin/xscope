import math
import os
import time
import logging
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from xscope.data_manager import RunDataManager

logger = logging.getLogger("xscope.diagnostics")


@dataclass
class DiagnosticResult:
    severity: str  # 'critical', 'warning', 'info'
    title: str
    message: str
    run_name: str
    epoch: Optional[int] = None
    suggested_action: Optional[str] = None

    def key(self) -> tuple:
        return (self.run_name, self.title, self.epoch)


@dataclass
class DiagnosticState:
    seen: set = field(default_factory=set)
    dismissed: set = field(default_factory=set)


DIAGNOSTIC_RULES: dict[str, Callable[[dict, RunDataManager], Optional[DiagnosticResult]]] = {}
CROSS_RUN_RULES: dict[str, Callable[[list[dict], RunDataManager], List[DiagnosticResult]]] = {}


def diagnostic_rule(name: str = "", description: str = "", enabled: bool = True):
    """Decorator to register a diagnostic rule."""
    def decorator(func):
        rule_name = name or func.__name__
        func._diag_meta = {
            "name": rule_name,
            "description": description,
            "enabled": enabled
        }
        DIAGNOSTIC_RULES[rule_name] = func
        return func
    return decorator


def cross_run_diagnostic_rule(name: str = "", description: str = "", enabled: bool = True):
    """Decorator to register a cross-run diagnostic rule."""
    def decorator(func):
        rule_name = name or func.__name__
        func._diag_meta = {
            "name": rule_name,
            "description": description,
            "enabled": enabled
        }
        CROSS_RUN_RULES[rule_name] = func
        return func
    return decorator


def _run_name(run: dict) -> str:
    exp_name = run.get('experiment_name', '')
    exp_num = run.get('experiment_number', '')
    if exp_num != "":
        return f"{exp_name} #{exp_num}"
    return str(exp_name) or run.get('run_path', 'unknown')


@diagnostic_rule(name="loss_divergence", description="Checks for NaN/Inf or validation loss divergence")
def check_loss_divergence(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    """Checks for NaN/Inf values or continuous validation loss divergence (overfitting)."""
    run_path = run.get('run_path', '')
    run_name = _run_name(run)

    _, records = data_manager.get_records(run_path, "metrics.jsonl")
    if not records:
        return None

    # Check for NaN / Inf
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                epoch_str = r.get('epoch', '?')
                return DiagnosticResult(
                    severity='critical',
                    title=f"Invalid Metric Value ('{k}')",
                    message=f"Metric '{k}' became NaN/Inf at epoch {epoch_str}.",
                    run_name=run_name,
                    epoch=r.get('epoch'),
                    suggested_action="Lower learning rate or enable gradient clipping.",
                )

    val_losses = []
    train_losses = []
    for r in records:
        if 'loss/val' in r and isinstance(r['loss/val'], (int, float)):
            val_losses.append((r.get('epoch'), r['loss/val']))
        if 'loss/train' in r and isinstance(r['loss/train'], (int, float)):
            train_losses.append((r.get('epoch'), r['loss/train']))

    if len(val_losses) >= 5:
        recent_5 = val_losses[-5:]
        
        # Check gradual increase (allowing plateaus)
        is_increasing = all(recent_5[i][1] <= recent_5[i + 1][1] for i in range(4))
        has_strict_increase = any(recent_5[i][1] < recent_5[i + 1][1] for i in range(4))
        
        if is_increasing and has_strict_increase:
            start_epoch = recent_5[0][0]
            end_epoch = recent_5[-1][0]
            # NOTE: returning here intentionally takes priority over the overfitting
            # gap check below — divergence is the stronger signal when both are present.
            return DiagnosticResult(
                severity='warning',
                title="Validation Loss Divergence",
                message=f"Validation loss increased consistently from epoch {start_epoch} to {end_epoch}.",
                run_name=run_name,
                epoch=end_epoch,
                suggested_action="Consider early stopping or increasing regularization (dropout/weight decay).",
            )

        # Check train/val gap (overfitting): train falling while val is flat/rising.
        if len(train_losses) >= 5:
            train_recent = train_losses[-5:]
            val_recent = val_losses[-5:]
            train_decreasing = train_recent[-1][1] < train_recent[0][1]
            val_increasing_or_flat = val_recent[-1][1] >= val_recent[0][1]
            
            if train_decreasing and val_increasing_or_flat:
                gap = val_recent[-1][1] - train_recent[-1][1]
                # Use max(0.05, ...) so near-zero val losses don't fire on noise.
                if gap > max(0.05, 0.1 * abs(val_recent[-1][1])):
                    return DiagnosticResult(
                        severity='warning',
                        title="Overfitting Detected",
                        message=f"Train loss decreasing while val loss isn't. Gap: {gap:.4f}",
                        run_name=run_name,
                        epoch=val_recent[-1][0],
                        suggested_action="Increase regularization or add early stopping.",
                    )

    # Check loss explosion (sudden spike)
    if len(val_losses) >= 3:
        prev_avg = (val_losses[-3][1] + val_losses[-2][1]) / 2
        if val_losses[-1][1] > prev_avg * 3 and val_losses[-1][1] > 0.01:
            return DiagnosticResult(
                severity='critical',
                title="Validation Loss Explosion",
                message=f"Val loss jumped from ~{prev_avg:.4f} to {val_losses[-1][1]:.4f}",
                run_name=run_name,
                epoch=val_losses[-1][0],
                suggested_action="Check for bad batch, reduce LR, enable gradient clipping.",
            )

    return None


@diagnostic_rule(name="class_collapse", description="Checks for class collapse and confusion matrix anomalies")
def check_class_collapse(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    """Checks for severe class collapse, low recall, and systematic confusion."""
    run_path = run.get('run_path', '')
    run_name = _run_name(run)

    _, records = data_manager.get_records(run_path, "matrix.jsonl")
    if not records:
        return None

    latest = records[-1]
    matrix = latest.get('matrix', [])
    labels = latest.get('labels', [])

    if not matrix or not isinstance(matrix, list):
        return None

    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 and isinstance(matrix[0], list) else 0
    if num_rows == 0 or num_cols == 0:
        return None

    # Validate matrix uniformity
    if not all(len(row) == num_cols for row in matrix):
        return None

    total_samples = 0
    col_sums = [0] * num_cols
    row_sums = [0] * num_rows
    
    for r_idx, row in enumerate(matrix):
        for c_idx, val in enumerate(row):
            if isinstance(val, (int, float)):
                col_sums[c_idx] += val
                row_sums[r_idx] += val
                total_samples += val

    if total_samples == 0:
        return None

    # 1. Check class collapse vs natural imbalance
    dominant_pred_ratio = max(col_sums) / total_samples
    dominant_true_ratio = max(row_sums) / total_samples
    if dominant_pred_ratio >= 0.85 and num_cols > 1 and dominant_pred_ratio > dominant_true_ratio + 0.15:
        col_idx = col_sums.index(max(col_sums))
        label_str = labels[col_idx] if col_idx < len(labels) else f"Class {col_idx}"
        return DiagnosticResult(
            severity='warning',
            title="Confusion Matrix Class Collapse",
            message=f"{round(dominant_pred_ratio * 100)}% of all predictions are assigned to '{label_str}'.",
            run_name=run_name,
            suggested_action="Check dataset class balance or adjust loss weights.",
        )

    # 2. Check per-class recall
    for i, row_sum in enumerate(row_sums):
        if row_sum > 0:
            recall = matrix[i][i] / row_sum
            if recall < 0.1 and num_cols > 2:
                label_str = labels[i] if i < len(labels) else f"Class {i}"
                return DiagnosticResult(
                    severity='warning',
                    title="Very Low Per-Class Recall",
                    message=f"Class '{label_str}' has recall of {round(recall*100)}%.",
                    run_name=run_name,
                    suggested_action="Add more training samples for this class or use focal loss.",
                )

    # 3. Check systematic class confusion
    max_off_diag = 0
    max_pair = None
    for i in range(num_rows):
        for j in range(num_cols):
            if i != j and isinstance(matrix[i][j], (int, float)):
                if matrix[i][j] > max_off_diag:
                    max_off_diag = matrix[i][j]
                    max_pair = (i, j)
                    
    # Use the true-class row sum as the denominator so this fires on imbalanced
    # datasets too — a minority class being 50%+ confused with another class is
    # just as significant even if it's only 5% of total_samples.
    if max_pair:
        true_row_sum = row_sums[max_pair[0]]
        if true_row_sum > 0 and max_off_diag > 0.5 * true_row_sum:
            true_label = labels[max_pair[0]] if max_pair[0] < len(labels) else f"Class {max_pair[0]}"
            pred_label = labels[max_pair[1]] if max_pair[1] < len(labels) else f"Class {max_pair[1]}"
            return DiagnosticResult(
                severity='info',
                title="Systematic Class Confusion",
                message=f"{round(max_off_diag/true_row_sum*100)}% of '{true_label}' misclassified as '{pred_label}'.",
                run_name=run_name,
                suggested_action="Inspect misclassified samples to understand feature overlap.",
            )

    return None


@diagnostic_rule(name="2d_anomaly", description="Checks for extreme coordinates or point collapse in 2D")
def check_2d_anomaly(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    """Checks for extreme coordinate anomalies or zero-variance point collapse in 2D spatial metrics."""
    run_path = run.get('run_path', '')
    run_name = _run_name(run)

    _, records = data_manager.get_records(run_path, "2d.jsonl")
    if not records:
        return None

    latest = records[-1]
    keys = [k for k in latest.keys() if k not in ('epoch', 'step')]

    for k in keys:
        pts = latest.get(k, [])
        if not isinstance(pts, list) or len(pts) < 3:
            continue

        valid_pts = [p for p in pts if isinstance(p, list) and len(p) >= 2 and p[0] is not None and p[1] is not None]
        if not valid_pts:
            continue

        xs = [p[0] for p in valid_pts]
        ys = [p[1] for p in valid_pts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Check extreme value anomaly relative to median absolute value
        abs_vals = [abs(v) for v in xs + ys if v != 0]
        med_abs = statistics.median(abs_vals) if abs_vals else 1.0
        
        if med_abs > 0 and any(abs(v) > med_abs * 1000 + 1e3 for v in xs + ys):
            return DiagnosticResult(
                severity='warning',
                title=f"2D Coordinate Anomaly ('{k}')",
                message=f"Extreme values detected in 2D points for '{k}' (x: [{min_x}, {max_x}], y: [{min_y}, {max_y}]).",
                run_name=run_name,
                suggested_action="Check spatial metric scaling or normalization layer outputs.",
            )

        # Check point collapse (variance near 0 for multi-point series)
        if len(valid_pts) >= 3:
            var_x = statistics.pvariance(xs)
            var_y = statistics.pvariance(ys)
            if var_x < 1e-10 and var_y < 1e-10:
                return DiagnosticResult(
                    severity='info',
                    title=f"2D Point Collapse ('{k}')",
                    message=f"All {len(valid_pts)} points for '{k}' collapsed near single coordinate ({min_x}, {min_y}).",
                    run_name=run_name,
                )

    return None


@diagnostic_rule(name="training_stall", description="Checks if accuracy/loss has stalled")
def check_training_stall(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    _, records = data_manager.get_records(run.get('run_path', ''), "metrics.jsonl")
    if len(records) < 10:
        return None
    
    recent = records[-10:]
    acc_key = next((k for k in recent[-1] if 'acc' in k.lower()), None)
    
    if acc_key:
        accs = [r[acc_key] for r in recent if isinstance(r.get(acc_key), (int, float))]
        if len(accs) >= 10:
            improvement = max(accs) - min(accs)
            if improvement < 0.001:
                return DiagnosticResult(
                    severity='info',
                    title='Training Stall',
                    message=f"Accuracy improved by only {improvement:.4f} over last 10 epochs.",
                    run_name=_run_name(run),
                    suggested_action="Increase learning rate or check if model has converged.",
                )
    return None


@diagnostic_rule(name="underfitting", description="Checks for high flat loss indicating underfitting")
def check_underfitting(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    _, records = data_manager.get_records(run.get('run_path', ''), "metrics.jsonl")
    if len(records) < 5:
        return None
    
    recent = records[-5:]
    train_losses = [r['loss/train'] for r in recent if 'loss/train' in r]
    if len(train_losses) >= 5:
        avg_loss = sum(train_losses) / len(train_losses)
        slope = (train_losses[-1] - train_losses[0]) / len(train_losses)
        # No absolute floor on avg_loss — a flat loss at any level (e.g. 0.6) can
        # indicate underfitting or a stuck optimiser and is worth surfacing.
        if abs(slope) < 0.001:
            return DiagnosticResult(
                severity='warning',
                title='Possible Underfitting',
                message=f"Train loss plateaued at {avg_loss:.4f} with near-zero slope.",
                run_name=_run_name(run),
                suggested_action="Increase model capacity, train longer, or reduce regularization.",
            )
    return None


@diagnostic_rule(name="data_leakage", description="Checks if val loss is consistently below train loss")
def check_data_leakage(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    _, records = data_manager.get_records(run.get('run_path', ''), "metrics.jsonl")
    pairs = [(r['loss/train'], r['loss/val']) for r in records 
             if 'loss/train' in r and 'loss/val' in r]
    if len(pairs) < 5:
        return None
    
    val_below_train = sum(1 for t, v in pairs[-5:] if v < t)
    if val_below_train >= 4:
        return DiagnosticResult(
            severity='warning',
            title='Validation Loss Below Training Loss',
            message="Val loss consistently lower than train loss — possible data leakage.",
            run_name=_run_name(run),
            suggested_action="Check for train/val data overlap or overly aggressive augmentation.",
        )
    return None


@diagnostic_rule(name="lr_too_high", description="Checks for loss oscillation indicating high LR")
def check_lr_too_high(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    # TODO: Do we know if the get_records would always give us the records? need to check.
    _, records = data_manager.get_records(run.get('run_path', ''), "metrics.jsonl")
    train_losses = [r['loss/train'] for r in records
                   if isinstance(r.get('loss/train'), (int, float))]
    if len(train_losses) < 6:
        return None
    
    recent = train_losses[-6:]
    diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
    sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
    
    if sign_changes >= 3:
        amplitude = max(recent) - min(recent)
        if amplitude > 0.1 * (sum(recent) / len(recent)):
            return DiagnosticResult(
                severity='warning',
                title='Loss Oscillation',
                message=f"Train loss oscillating with amplitude {amplitude:.4f} over last 6 epochs.",
                run_name=_run_name(run),
                suggested_action="Learning rate may be too high. Consider reducing by 10×.",
            )
    return None


@diagnostic_rule(name="2d_prediction_divergence", description="Checks if 2D predictions drift from ground truth")
def check_2d_prediction_divergence(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    _, records = data_manager.get_records(run.get('run_path', ''), "2d.jsonl")
    if len(records) < 3:
        return None
    
    errors = []
    for r in records:
        true_pts = r.get('data/true', [])
        pred_pts = r.get('data/pred', [])
        if len(true_pts) == len(pred_pts) and len(true_pts) > 0:
            # Include both X and Y axes in the squared error so spatial drift
            # in either dimension is captured, not just the Y component.
            total_err = sum(
                (t[0] - p[0])**2 + (t[1] - p[1])**2
                for t, p in zip(true_pts, pred_pts)
                if isinstance(t, list) and isinstance(p, list) and len(t) >= 2 and len(p) >= 2
            )
            errors.append((r.get('epoch'), total_err / len(true_pts)))
    
    if len(errors) >= 3:
        first_err = errors[0][1]
        last_err = errors[-1][1]
        if last_err > first_err * 2 and last_err > 0.01:
            return DiagnosticResult(
                severity='warning',
                title='2D Prediction Divergence',
                message=f"MSE grew from {first_err:.4f} (epoch {errors[0][0]}) to {last_err:.4f} (epoch {errors[-1][0]}).",
                run_name=_run_name(run),
                epoch=errors[-1][0],
                suggested_action="Model predictions are diverging from ground truth over time.",
            )
    return None


@diagnostic_rule(name="epoch_progress", description="Checks if training has unexpectedly stalled/crashed")
def check_epoch_progress(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    _, records = data_manager.get_records(run.get('run_path', ''), "metrics.jsonl")
    if len(records) < 2:
        return None
    
    last_epoch = records[-1].get('epoch')
    if last_epoch is None:
        return None
    
    target_epochs = run.get('target_epochs') or run.get('epochs')
    if target_epochs and isinstance(target_epochs, int) and last_epoch < target_epochs:
        filepath = os.path.join(run['run_path'], "metrics.jsonl")
        if os.path.isfile(filepath):
            age = time.time() - os.path.getmtime(filepath)
            if age > 3600:  # 1 hour stale
                return DiagnosticResult(
                    severity='critical',
                    title='Training Appears Stalled',
                    message=f"Last epoch {last_epoch}/{target_epochs}, file unchanged for {round(age/60)}min.",
                    run_name=_run_name(run),
                    epoch=last_epoch,
                    suggested_action="Check if training process crashed or was killed.",
                )
    return None


@diagnostic_rule(name="metric_regression", description="Checks for sudden accuracy drops")
def check_metric_regression(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    _, records = data_manager.get_records(run.get('run_path', ''), "metrics.jsonl")
    if len(records) < 3:
        return None
    
    acc_key = next((k for k in records[-1] if 'acc' in k.lower()), None)
    if not acc_key:
        return None
    
    accs = [(r.get('epoch'), r[acc_key]) for r in records if isinstance(r.get(acc_key), (int, float))]
    if len(accs) < 3:
        return None
    
    best_so_far = max(a for _, a in accs[:-1])
    current = accs[-1][1]
    if current < best_so_far - 0.1:  # 10% absolute drop
        return DiagnosticResult(
            severity='warning',
            title='Accuracy Regression',
            message=f"Accuracy dropped from best {best_so_far:.4f} to {current:.4f}.",
            run_name=_run_name(run),
            epoch=accs[-1][0],
            suggested_action="Check for LR schedule warmup/restart or data shuffling issues.",
        )
    return None


@diagnostic_rule(name="missing_metric_files", description="Checks for missing metrics files")
def check_missing_metric_files(run: dict, data_manager: RunDataManager) -> Optional[DiagnosticResult]:
    run_path = run.get('run_path', '')
    filepath = os.path.join(run_path, 'metrics.jsonl')
    if not os.path.isfile(filepath):
        return DiagnosticResult(
            severity='info',
            title='Missing Metrics File',
            message="'metrics.jsonl' not found in run directory.",
            run_name=_run_name(run),
        )
    return None


@cross_run_diagnostic_rule(name="outlier_val_loss", description="Detects runs with val loss much worse than siblings")
def check_outlier_val_loss(selected_runs: list[dict], data_manager: RunDataManager) -> List[DiagnosticResult]:
    results = []
    final_losses = []
    
    for run in selected_runs:
        _, records = data_manager.get_records(run.get('run_path', ''), "metrics.jsonl")
        if records:
            last_val = records[-1].get('loss/val')
            if isinstance(last_val, (int, float)) and not math.isnan(last_val):
                final_losses.append((run, last_val))
                
    if len(final_losses) < 3:
        return results
        
    losses = [l for _, l in final_losses]
    median_loss = statistics.median(losses)
    
    for run, loss in final_losses:
        if loss > median_loss * 1.5 and median_loss > 0.01:
            results.append(DiagnosticResult(
                severity='info',
                title='Outlier Validation Loss',
                message=f"Val loss is {loss:.4f}, >1.5x the median of selected runs ({median_loss:.4f}).",
                run_name=_run_name(run),
                suggested_action="Compare hyperparameters and configurations with better-performing runs.",
            ))
            
    return results


def run_diagnostics_for_runs(
    selected_runs: list[dict], 
    data_manager: RunDataManager, 
    state: Optional[DiagnosticState] = None
) -> List[DiagnosticResult]:
    """Evaluates all registered diagnostic rules across the specified selected runs."""
    results: List[DiagnosticResult] = []
    
    # Per-run rules
    for run in selected_runs:
        for rule_name, rule in DIAGNOSTIC_RULES.items():
            if not rule._diag_meta.get("enabled", True):
                continue
            try:
                res = rule(run, data_manager)
                if res:
                    results.append(res)
            except Exception:
                logger.exception("Error in diagnostic rule '%s'", rule_name)
                
    # Cross-run rules
    for rule_name, rule in CROSS_RUN_RULES.items():
        if not rule._diag_meta.get("enabled", True):
            continue
        try:
            cross_results = rule(selected_runs, data_manager)
            results.extend(cross_results)
        except Exception:
            logger.exception("Error in cross-run diagnostic rule '%s'", rule_name)

    # Deduplicate and filter dismissed
    if state:
        filtered_results = []
        for res in results:
            k = res.key()
            if k in state.dismissed:
                continue
            if k not in state.seen:
                state.seen.add(k)
                filtered_results.append(res)
            else:
                # If it's a new occurrence of a seen issue (e.g. epoch changed), keep it
                # The key includes epoch, so same issue at new epoch will naturally pass
                filtered_results.append(res)
        results = filtered_results

    # Sort by severity
    SEVERITY_ORDER = {'critical': 0, 'warning': 1, 'info': 2}
    results.sort(key=lambda r: (SEVERITY_ORDER.get(r.severity, 3), r.run_name))
    
    return results