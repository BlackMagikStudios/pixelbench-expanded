# Results format

`pixelbench run` writes a single JSON file. The machine-readable schema is
[`results/schema.json`](../results/schema.json); this is the human version.

## Top level

```jsonc
{
  "pixelbench_version": "0.1.0",
  "suite": "v1",                  // distortion suite version
  "severity": 1.0,
  "seed": 1234,
  "dataset": { "path": "...", "n_images": 90 },
  "methods": ["fixer", "snapper", "pixeldetector", "unfake", "naive"],
  "n_inputs": 990,                // images x categories
  "n_scored": 990,
  "wall_s": 610.0,
  "aggregate": { ... },           // precomputed summary (below)
  "samples": [ ... ]              // one entry per (image, category)
}
```

## `samples[]`

One entry per distorted input, holding every method's scores on it:

```jsonc
{
  "image": "frog.png",
  "category": "soft_bilinear",
  "severity": 1.0,
  "gt_cols": 32, "gt_rows": 32,        // true native size
  "dist_w": 192, "dist_h": 192,        // distorted input size
  "methods": {
    "fixer":   { "exact": 1, "within1": 1, "rel_err": 0.0,
                 "delta_e": 1.4, "psnr": 34.2,
                 "pixel_match": 0.97, "grid_align": 1.0,
                 "ncolor_err": 0.1, "palette_de": 2.1, "sec": 1.8 },
    "snapper": { ... },
    "unfake":  { "error": "…", "sec": 0.3 }   // a failed run keeps its error
  }
}
```

Skipped inputs (too small after distortion, decode error) appear as
`{ "image", "category", "skip": "reason" }` with no `methods`.

The metric fields are defined in [METRICS.md](METRICS.md). A method that fails
on an input records `{"error": "...", "sec": ...}` instead of scores, so one bad
input never sinks the whole run.

## `aggregate`

Precomputed so consumers don't have to re-reduce `samples`:

```jsonc
{
  "categories": ["clean_nn", "fractional", ...],
  "directions": { "exact": "higher", "rel_err": "lower", ... },
  "by_method": {
    "fixer": {
      "clean_nn":   { "n": 90, "exact": 0.98, "delta_e": 0.9, ... },
      "soft_bilinear": { ... },
      "overall":    { "n": 990, "exact": 0.68, "grid_align": 0.89, ... }
    },
    ...
  }
}
```

Rate fields are averaged; error fields (`rel_err`, `delta_e`, `psnr`,
`palette_de`, `sec`) use the median. `directions` tells you which way is better
for every field, so tooling can rank without hard-coding metric names.

## Comparability

Two runs are directly comparable when they share the same `suite`, `severity`,
and **dataset**. The distortion for a given `(image, category)` is deterministic
across machines, so re-running the same dataset reproduces the same inputs.
