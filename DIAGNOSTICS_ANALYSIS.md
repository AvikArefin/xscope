# xscope Diagnostics

xscope inspects the selected runs and reports problems in the **Diagnostics** panel (shield icon in the header).
The code lives in [`xscope/diagnostics/`](xscope/diagnostics), and every rule has a scenario test in
[`tests/test_diagnostics.py`](tests/test_diagnostics.py).

## Design principles

1. **No hardcoded metric names.** A metric's role is inferred from its name (`series.classify_metric`):
   - **split:** `train`/`training`/`trn` vs. `val`/`valid`/`validation`/`dev`/`eval`/`test`
   - **kind:**
     - loss-like: `loss`, `mse`, `mae`, `nll`, `ce`, `error`, `perplexity`, …
     - score-like: `acc`, `f1`, `precision`, `recall`, `auc`, `iou`, `dice`, …
     - `lr` / `learning_rate`
     - `grad…`
   - **pairing:** train and val metrics are paired by their name with the split removed. So `loss/train` ↔ `loss/val`, `train_loss` ↔ `val_loss` and `trainAcc` ↔ `valAcc` all work.
2. **Scale-free thresholds.** Every threshold is relative to the curve's own noise (a MAD-based estimate from first
   differences) or to how much the curve improved. The same rules work for a loss of 2.3 or 0.002.
3. **Smoothed decisions, raw advice.** Decisions use a rolling-median curve or averages of several points, so a single
   noisy epoch can't trigger them. Advice such as "keep the checkpoint from epoch N" points at the best *logged* epoch.
4. **Rules know about each other.** A run is never called both "overfitting" and "plateaued". Spikes aren't reported on a
   run that already diverged. Accuracy drops on validation data are folded into the overfitting finding.
5. **Severity means something:**
   - **critical:** the run is broken (NaN, divergence, exploding gradients).
   - **warning:** something to act on.
   - **info:** context, possibly benign.

## Validation

- **False alarms:** 4,500 random *healthy* runs (exponential and power-law curves, 8–400 epochs, 0.2–6% noise, val loss
  ±15% from train) produced warnings on 2 runs (0.04%). Both were 8-epoch runs whose total improvement was smaller than
  their noise.
- **Detection rates** on 40 synthetic runs per cell:

  | Problem | 0.5% noise | 2% noise | 5% noise |
  |---|---|---|---|
  | Overfitting | 40/40 | 31/40 | 28/40 |
  | Divergence | 40/40 | 40/40 | 40/40 |
  | Not learning | 40/40 | 38/40 | 39/40 |
  | Accuracy regression | 40/40 | 39/40 | 40/40 |

## Rule catalog

### Scalar metrics (`metrics.jsonl`) — `rules_scalar.py`

| Rule | Severity | Fires when |
|---|---|---|
| `nonfinite_values` | critical | Any metric is NaN/Inf (reports the first epoch per metric). |
| `metric_integrity` | warning / info | The epoch counter goes backwards (several sessions appended into one file). A loss, score or grad metric is identical in ≥ 8 records. A bounded score (accuracy, F1, IoU, …) falls outside [0, 100]. A loss that can't be negative (MSE, CE, …) is negative. A metric stopped being logged (allowing for metrics that are logged only every N epochs). |
| `empty_run` | info | The run directory has no metrics, 2D or matrix records. |
| `loss_divergence` | critical / warning | Critical: a loss climbed back above its starting value. Warning: it gave back ≥ 30% of its improvement. |
| `loss_spikes` | warning / info | A jump of > 8σ **and** > 30% of the local level. Info if the loss later recovered. Mentions a coinciding learning-rate increase when an `lr` metric is logged, and suggests a warmup when an early spike happens at full learning rate. |
| `training_instability` | warning | Epoch-to-epoch noise at least doubled (absolute) between the first and last third, and grew ≥ 2.5× relative to the loss level. Needs ≥ 45 points. |
| `not_learning` | warning | Averages of the first and last ~20% of epochs differ by < 2% (losses) or < 2% of the remaining headroom (bounded scores), or by less than noise. A clear drop from the very first point counts as learning. |
| `plateau` | info | The loss stayed within 2% of its total improvement for ≥ 25% of the run. Suggests stopping earlier, and notes when the logged LR was never decayed during the plateau. |
| `gradient_health` | critical / warning | For metrics named `grad…`: recent norm ≥ 10× its early level (exploding), a single spike ≥ 100×, or a fall to < 1/1000 (vanishing). |
| `overfitting` | warning | The mean of the last ~20% of val-loss points exceeds the best region by more than 3.5 standard errors (and ≥ 2% of its level). Train loss must be learning. If train kept falling, the finding is "Overfitting", otherwise "Validation loss regressing". Notes whether val accuracy also dropped, or held (over-confidence or calibration drift). |
| `val_below_train` | info | Val loss is > 5% below train loss in ≥ 4 of the last 5 epochs. Explains the usual benign causes (dropout, augmentation) before suggesting a leakage check. |
| `score_regression` | warning | An accuracy-like metric dropped from its peak by > 4σ and > 15% of its total gain. Skipped for validation scores already covered by the overfitting finding. |
| `stopped_while_improving` | info | The run finished (per `status.json`, or no writes for ≥ 30× its logging interval in older runs), while a line fitted to the last 10% of epochs still shows > 5% of the total improvement, beyond noise. Points to an under-trained run. Cosine/step schedules stay quiet because their LR is ~0 at the very end. |
| `score_gap` | warning / info | A train/val pair of accuracy-like metrics differs by > 10 points (bounded scores) or > 10% (others), averaged over the last epochs. Notes when train accuracy is saturated (memorization). Info instead of warning when the loss overfitting finding already fired. |

### Run lifecycle — `rules_lifecycle.py`

These rules need `status.json` and per-record `_time`, both written by current mtrick. Older runs are skipped.

| Rule | Severity | Fires when |
|---|---|---|
| `run_status` | critical / warning / info | `crashed`: reports the exception and last epoch, with out-of-memory-specific advice. `interrupted`: info. `running` but silent for ≥ 10× the usual logging interval (≥ 10 min): the process was likely killed (OOM killer, preemption) or hung. |
| `training_slowdown` | warning | The median time between records in the last third is > 2× the first third (and > 1 s longer). |
| `memory_creep` | warning | A memory metric (`mem`, `memory`, `rss`, `vram`; e.g. mtrick's `sys/peak_rss_gb`) grows across all three thirds of the run and ends > 30% above its start. Memory that ramps up and then levels off stays quiet. |

### Calibration (`calibration.jsonl`) — `rules_calibration.py`

| Rule | Severity | Fires when |
|---|---|---|
| `calibration` | warning / info | Expected calibration error ≥ 10% (warning) or ≥ 5% (info) on ≥ 50 predictions. Says whether the model is over- or under-confident, and whether calibration has worsened since the first step. |

### Confusion matrices (`matrix.jsonl`) — `rules_matrix.py`

Rows are true classes and columns are predictions. Statistical rules need ≥ 20 samples in total, and per-class rules
need ≥ 5 samples of that class. Lists show the worst 3 items, so 100-class matrices stay readable.

| Rule | Severity | Fires when |
|---|---|---|
| `matrix_integrity` | warning / info | The matrix isn't square, has non-numeric or negative entries, or has the wrong number of labels. Classes have no true samples. The evaluation-set size changes between steps. |
| `matrix_quality` | warning / info | Each check below runs only if the ones above it didn't fire (see the list after this table). |
| `matrix_class_regression` | warning | A class's recall or precision fell ≥ 20 points (and below 80% of its best) compared with an earlier step. |
| `matrix_scalar_mismatch` | warning | A logged accuracy metric (`acc`, `accuracy`, `top1`; val or unsplit, fraction or percent) differs by > 2 points from the confusion-matrix accuracy at the same step. Usually different eval subsets or averaged per-batch accuracies. |

The checks inside `matrix_quality`, in order:

1. **Prediction collapse or bias:** one class receives ≥ 50% of predictions and ≥ 30 points more than its true share.
2. **No better than a trivial baseline:** accuracy is at most 2 points above the majority-class share (or chance level).
3. **Hidden minority failure:** accuracy beats balanced accuracy by ≥ 10 points on data with ≥ 3× class imbalance.
4. **Very low recall:** individual classes with recall ≤ 10%, or below half of the balanced accuracy.
5. **Imbalanced evaluation set:** the largest class has ≥ 10× the samples of the smallest.
6. **Over-predicted classes:** classes predicted ≥ 1.5× as often as they occur, with precision < 50%.
7. **Systematic confusions:** a true class predicted as one specific other class ≥ 30% of the time. Mutual pairs are shown as ↔.

### 2D points (`2d.jsonl`) — `rules_2d.py`

Ground-truth and prediction series are paired by name:
- **truth tokens:** `true`, `gt`, `target`, `label`, `actual`, …
- **prediction tokens:** `pred`, `prediction`, `output`, `est`, …

For example, `data/true` ↔ `data/pred` and `traj_gt` ↔ `traj_pred`.

| Rule | Severity | Fires when |
|---|---|---|
| `points_2d_integrity` | critical / warning / info | NaN/Inf coordinates, points that aren't `[x, y]`, or truth and prediction with different point counts. |
| `prediction_2d_quality` | warning / info | Checked in this order: outliers (≤ 20% of points with > 10× the median error; excluded from the next checks); collapse (prediction spread < 20% of truth spread); over-dispersion (> 3×, usually a scale mismatch); systematic offset (mean residual ≥ 70% of RMSE). When the error is non-trivial, it also checks: error that accumulates along the point sequence (drift; correlation with index ≥ 0.6 and last quarter ≥ 2.5× first); error concentrated on one axis (≥ 3× after normalizing by each axis's spread); and per-axis scale bias (a scaled copy of the truth, factor outside 0.85–1.18). |
| `prediction_2d_trend` | warning / info | RMSE rose ≥ 1.5× above its best epoch, or never improved on the first record. |
| `series_2d_collapse` | info | An unpaired series has all its points identical. |

### Across selected runs — `rules_cross.py`

| Rule | Severity | Fires when |
|---|---|---|
| `outlier_runs` | info | Among ≥ 3 runs, a run's final val loss is above the median by more than 3 robust σ and 50%. Lists the config values where it differs from what all the other runs share. |
| `inconclusive_comparison` | info | The best and runner-up runs, by val score (else val loss), differ by less than twice their combined standard error. The noise is estimated from second differences of each curve's tail, so smooth curves aren't mistaken for noisy ones. |
| `config_attribution` | info | With ≥ 2 runs with warnings and ≥ 2 healthy runs (all with `config.json`), a setting cleanly separates them (e.g. `lr ≥ 0.1` vs `≤ 0.001`). Identity-like keys (seed, name, path, …) are ignored. Presented as a lead, not proof. |
| `config_consistency` | info | The same config key is a number in one run and a string in another (e.g. `0.001` vs `"1e-3"`). |
| `metric_name_mismatch` | info | Runs log the same metric under different names (e.g. `loss/val` vs `val_loss`), so their curves end up in separate charts. |
| `eval_set_mismatch` | info | Latest confusion matrices of different runs have different sample counts. |
| `uncommitted_code` | info | Runs were launched from a dirty git tree (commit marked `*`). |

Crashed and interrupted runs are left out of `outlier_runs` and `inconclusive_comparison`, because their "final" values come from partway through training.

## Adding a rule

```python
from xscope.diagnostics import DiagnosticResult, RunContext, diagnostic_rule

@diagnostic_rule(name="my_rule", description="One-line description")
def check_my_rule(ctx: RunContext):
    t = ctx.trend("loss/train")      # oriented, smoothed series (or None)
    if t is None:
        return None
    ...
    return DiagnosticResult('warning', "Title", "What happened, with numbers.", ctx.name,
                            suggested_action="What to do about it.", metric="loss/train")
```

A rule may return `None`, one result, or a list. `RunContext` caches records, metric roles and trends, so rules are
cheap to combine. Exceptions inside a rule are logged and never break the other rules. Cross-run rules use
`@cross_run_diagnostic_rule` and receive a list of contexts.
