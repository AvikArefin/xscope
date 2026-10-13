"""Metric-name interpretation and robust series statistics shared by the diagnostic rules.

Every threshold built on these helpers is relative — to a series' own noise level
or to how much it improved — so the rules behave the same for a cross-entropy
loss starting at 2.3 and an MSE loss starting at 0.002.
"""
import math
import re
import statistics
from dataclasses import dataclass

TRAIN_TOKENS = {'train', 'training', 'trn'}
VAL_TOKENS = {'val', 'valid', 'validation', 'dev', 'eval', 'evaluation', 'test', 'holdout'}

LOWER_BETTER_TOKENS = {
    'loss', 'error', 'err', 'mse', 'mae', 'rmse', 'nll', 'ce', 'xent', 'crossentropy',
    'cost', 'perplexity', 'ppl', 'wer', 'cer', 'l1', 'l2', 'huber', 'kl', 'bce', 'mape', 'smape',
}
HIGHER_BETTER_TOKENS = {
    'acc', 'accuracy', 'top1', 'top5', 'f1', 'fscore', 'precision', 'recall', 'auc', 'auroc',
    'ap', 'map', 'iou', 'miou', 'dice', 'r2', 'bleu', 'rouge', 'score', 'sensitivity',
    'specificity', 'mcc', 'kappa',
}
# Higher-is-better metrics that live in [0, 1] (or [0, 100] when logged as percentages).
BOUNDED_TOKENS = {
    'acc', 'accuracy', 'top1', 'top5', 'f1', 'fscore', 'precision', 'recall', 'auc', 'auroc',
    'ap', 'map', 'iou', 'miou', 'dice', 'sensitivity', 'specificity',
}
# Losses that cannot be negative by construction.
NON_NEGATIVE_LOSS_TOKENS = {
    'mse', 'mae', 'rmse', 'ce', 'xent', 'crossentropy', 'bce', 'l1', 'l2', 'huber',
    'perplexity', 'ppl', 'wer', 'cer', 'mape', 'smape',
}
MEMORY_TOKENS = {'mem', 'memory', 'rss', 'vram'}
X_AXIS_KEYS = {'epoch', 'step', 'timestamp', 'time', 'wall_time', 'iteration', 'iter', 'global_step'}


def tokenize(key: str) -> list[str]:
    """Splits 'loss/train', 'train_loss', 'valLoss' etc. into lowercase word tokens."""
    key = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', key)
    return [t for t in re.split(r'[^a-z0-9]+', key.lower()) if t]


@dataclass(frozen=True)
class MetricInfo:
    key: str
    kind: str          # 'loss' | 'score' | 'lr' | 'grad' | 'memory' | 'other'
    split: str | None  # 'train' | 'val' | None
    base: str          # name without split tokens, used to pair train/val metrics
    bounded: bool      # score expected in [0, 1] or [0, 100]
    non_negative: bool # loss that can never be negative

    @property
    def lower_is_better(self) -> bool:
        return self.kind != 'score'


def classify_metric(key: str) -> MetricInfo:
    tokens = tokenize(key)
    token_set = set(tokens)

    if token_set & TRAIN_TOKENS:
        split = 'train'
    elif token_set & VAL_TOKENS:
        split = 'val'
    else:
        split = None

    if 'lr' in token_set or 'learningrate' in token_set or {'learning', 'rate'} <= token_set:
        kind = 'lr'
    elif token_set & MEMORY_TOKENS:
        kind = 'memory'
    elif any(t in LOWER_BETTER_TOKENS or t.startswith('loss') for t in tokens):
        kind = 'loss'
    elif any(t.startswith('grad') for t in tokens):
        kind = 'grad'
    elif any(t in HIGHER_BETTER_TOKENS or t.startswith('acc') for t in tokens):
        kind = 'score'
    else:
        kind = 'other'

    base = '/'.join(t for t in tokens if t not in TRAIN_TOKENS | VAL_TOKENS) or key.lower()
    bounded = kind == 'score' and any(t in BOUNDED_TOKENS or t.startswith('acc') for t in tokens)
    non_negative = kind == 'loss' and bool(token_set & NON_NEGATIVE_LOSS_TOKENS)
    return MetricInfo(key, kind, split, base, bounded, non_negative)


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_finite_number(v) -> bool:
    return is_number(v) and math.isfinite(v)


def fmt(v: float) -> str:
    """Compact number formatting for messages (0.0123, 1.23e+04, 87.5)."""
    if v is None:
        return '?'
    if v == 0:
        return '0'
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.3g}"
    return f"{v:.4g}"


def robust_sigma(values: list[float]) -> float:
    """Standard deviation estimated from the median absolute deviation (outlier-resistant)."""
    if len(values) < 2:
        return 0.0
    med = statistics.median(values)
    return 1.4826 * statistics.median(abs(v - med) for v in values)


def noise_sigma(values: list[float]) -> float:
    """Per-point noise of a series, estimated from its first differences.

    Subtracting the median difference removes a steady trend, so a smoothly
    decreasing loss has ~0 noise while a jittery one does not.
    """
    if len(values) < 3:
        return 0.0
    diffs = [b - a for a, b in zip(values, values[1:])]
    return robust_sigma(diffs) / math.sqrt(2)


def local_noise_sigma(values: list[float]) -> float:
    """Per-point noise from second differences, which cancel a locally linear trend.

    Unlike `noise_sigma`, this is ~0 for any smooth curve (no bias from curvature),
    at the cost of being noisier on very short series.
    """
    if len(values) < 4:
        return 0.0
    second = [values[i + 1] - 2 * values[i] + values[i - 1] for i in range(1, len(values) - 1)]
    return robust_sigma(second) / math.sqrt(6)


def rolling_median(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 3:
        return list(values)
    half = window // 2
    n = len(values)
    return [statistics.median(values[max(0, i - half):min(n, i + half + 1)]) for i in range(n)]


def smoothing_window(n: int) -> int:
    if n < 5:
        return 1
    if n < 15:
        return 3
    if n < 60:
        return 5
    return min(51, (n // 12) | 1)


def downsample(xs: list[float], ys: list[float], max_points: int = 400) -> tuple[list[float], list[float]]:
    """Bucket-averages long series so trend analysis stays fast on per-step logs."""
    n = len(ys)
    if n <= max_points:
        return xs, ys
    out_x, out_y = [], []
    for b in range(max_points):
        lo, hi = b * n // max_points, (b + 1) * n // max_points
        out_x.append(xs[hi - 1])
        out_y.append(sum(ys[lo:hi]) / (hi - lo))
    return out_x, out_y


@dataclass
class Trend:
    """A metric oriented so that lower is always better, with smoothed summary points.

    Scores (accuracy, F1, …) are negated internally; `value()` converts back to the
    units the user logged.
    """
    key: str
    sign: int               # +1 for lower-is-better metrics, -1 for scores
    xs: list[float]
    raw: list[float]        # oriented, full resolution
    smooth: list[float]     # oriented, downsampled + rolling median
    smooth_xs: list[float]
    sigma: float            # per-point noise of the oriented raw series

    @property
    def n(self) -> int:
        return len(self.raw)

    @property
    def start(self) -> float:
        return self.smooth[0]

    @property
    def last(self) -> float:
        return self.smooth[-1]

    @property
    def best_index(self) -> int:
        return min(range(len(self.smooth)), key=lambda i: self.smooth[i])

    @property
    def best(self) -> float:
        return self.smooth[self.best_index]

    @property
    def best_x(self) -> float:
        return self.smooth_xs[self.best_index]

    @property
    def improvement(self) -> float:
        """How far the smoothed series got below its starting value (>= 0)."""
        return max(0.0, self.start - self.best)

    @property
    def window_k(self) -> int:
        """Number of raw points averaged for 'start' and 'end' comparisons."""
        return max(3, self.n // 5)

    def head_mean(self) -> float:
        return statistics.fmean(self.raw[:self.window_k])

    def tail_mean(self) -> float:
        return statistics.fmean(self.raw[-self.window_k:])

    def best_region(self) -> list[int]:
        """Raw indices covered by the smoothing window around the smoothed optimum."""
        b = self.best_index
        h = smoothing_window(len(self.smooth)) // 2
        lo = self.smooth_xs[b - h - 1] if b - h - 1 >= 0 else None
        hi = self.smooth_xs[min(len(self.smooth) - 1, b + h)]
        region = [i for i in range(self.n) if (lo is None or self.xs[i] > lo) and self.xs[i] <= hi]
        return region or list(range(self.n))

    def checkpoint(self) -> tuple[float, float]:
        """(x, oriented value) of the best *logged* point near the smoothed optimum.

        Decisions use the smoothed curve so noise can't trigger them, but advice like
        "keep the checkpoint from epoch N" must point at an epoch that was actually best.
        """
        i = min(self.best_region(), key=lambda j: self.raw[j])
        return self.xs[i], self.raw[i]

    def value(self, oriented: float) -> float:
        return oriented * self.sign

    def value_near(self, x: float) -> float:
        """Smoothed oriented value at the point closest to x."""
        i = min(range(len(self.smooth_xs)), key=lambda j: abs(self.smooth_xs[j] - x))
        return self.smooth[i]


def build_trend(key: str, xs: list[float], ys: list[float], lower_is_better: bool) -> Trend:
    sign = 1 if lower_is_better else -1
    raw = [sign * y for y in ys]
    dxs, dys = downsample(xs, raw)
    smooth = rolling_median(dys, smoothing_window(len(dys)))
    return Trend(key, sign, list(xs), raw, smooth, dxs, noise_sigma(raw))


def format_x(x) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)
