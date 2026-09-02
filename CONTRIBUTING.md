# Contributing to pixel-bench

pixel-bench is meant to be the shared yardstick for pixel-art reconstruction,
which only works if it is easy to add the thing you care about: a new method to
benchmark, a new metric to measure, or a new distortion to survive. Each is a
single self-contained file. You never touch the runner.

## Add a reconstruction method

Create `pixelbench/methods/yourtool.py`:

```python
from .base import Method, Reconstruction, register

@register
class YourTool(Method):
    name = "yourtool"                 # unique, lowercase
    requires = ("yourtool_pkg",)      # third-party imports; omit if none.
                                      # missing deps -> method auto-skipped
    def reconstruct(self, image):     # image: HxWx4 uint8 (the distorted input)
        # ... your algorithm ...
        native = ...                  # recovered 1x pixel art, uint8 HxWx3 or HxWx4
        return Reconstruction(native=native)   # cols/rows inferred from native
```

That's it - `pixelbench list-methods` and every run now include it. Notes:

- Return `Reconstruction(native=..., cols=c, rows=r)` if you want to report a
  predicted size that differs from `native.shape` (rare).
- A **detect-only** method can return `Reconstruction(cols=c, rows=r)` with no
  `native`; it is scored on resolution and grid alignment only.
- Put third-party backends in `requires` so people without them still run the
  rest of the benchmark. Wrap heavy imports inside `reconstruct`.
- Adapters for external tools are welcome (see `methods/unfake.py` for the
  optional-dependency pattern).

## Add a metric

Create `pixelbench/metrics/yourmetric.py`:

```python
from .base import Metric, register

@register
class YourMetric(Metric):
    name = "yourmetric"
    fields = {"your_score": "higher"}    # each field: "higher" or "lower" is better
    def score(self, sample, rec):
        # sample.gt, sample.distorted, sample.U, sample.V,
        # sample.gt_cols/gt_rows, sample.category are all available.
        # rec.native / rec.cols / rec.rows is the method's output.
        if rec.native is None:
            return {}                    # skip cleanly when not applicable
        return {"your_score": ...}
```

`docs/METRICS.md` documents the existing signals and the aggregation rules.

## Add a distortion category

Append a `Category` to `CATEGORIES` in `pixelbench/distort/suite.py`
(append-only, or bump `SUITE_VERSION` - never silently change an existing
category, it breaks comparability). See `docs/DISTORTIONS.md`.

## Ground rules

- **Fairness:** a method sees only the distorted RGBA image - never the ground
  truth, the spec, or `U`/`V`. Metrics get the ground truth; methods do not.
- **No dataset:** do not commit pixel-art corpora. Users bring their own native
  1x art. Tiny synthetic fixtures for tests are fine.
- **Determinism:** avoid wall-clock time and unseeded randomness in engine,
  suite, and metrics so results reproduce across machines.
- **Keep the core light:** the base install is numpy/scipy/opencv/Pillow. New
  heavy dependencies belong behind a method's `requires`.

## Sanity check your addition

```bash
pixelbench list-methods         # your method shows up (and is 'available')
pixelbench list-metrics
pixelbench run --data ./some_1x_art --limit 5 --methods yourtool --workers 2
```
