# Metrics

This file documents the metrics inherited from the original PixelBench
protocol. The complementary fork metrics are documented separately in
[EXPANDED_METRICS.md](EXPANDED_METRICS.md), and the protocol distinction is
defined in [KNOWN_SCALE_PROTOCOL.md](KNOWN_SCALE_PROTOCOL.md).

pixel-bench scores each reconstruction on several **independent signals** rather
than one blended number, so a method's specific strengths and weaknesses are
visible. All metrics compare a method's recovered native image (and/or its
predicted `cols x rows`) against the exact ground truth from the distortion
engine.

Each metric declares, per field, whether **higher** or **lower** is better.

## resolution

Did the method recover the true native size? This is the hardest and most
consequential signal - every pixel downstream depends on it.

| Field | Better | Meaning |
|---|---|---|
| `exact` | higher | both `cols` and `rows` exactly right (0/1) |
| `within1` | higher | both within ±1 cell |
| `within5pct` | higher | both within 5% (min 1 cell) |
| `rel_err` | lower | `mean(|Δcols|/W, |Δrows|/H)` |

## color

Are the reconstructed pixels the right colour? The reconstruction is
nearest-resampled to the true native size, then every reconstructed pixel is
compared to the intended original pixel.

| Field | Better | Meaning |
|---|---|---|
| `delta_e` | lower | mean CIELAB colour difference (dE76); ~1 is a just-noticeable difference |
| `psnr` | higher | peak signal-to-noise ratio (dB) |

> **Confound - read before trusting raw ΔE.** Colour error over *all* samples
> flatters methods that **over-segment**. Predicting more cells than the truth
> preserves colour detail that survives the nearest downsize, so a structurally
> wrong result can still post a low per-pixel colour error. The report therefore
> also shows ΔE **conditioned on the exact-resolution subset** - a like-for-like
> comparison of colour quality when the grid is actually right.

## placement

Are the pixels in the right place, and did the method find the real grid? Both
signals are distinct from colour.

| Field | Better | Meaning |
|---|---|---|
| `pixel_match` | higher | share of native pixels whose colour exactly matches the intended original pixel (within a tiny tolerance) after nearest-resampling to true size |
| `grid_align` | higher | F1 of the method's implied grid lines against the **true** grid cut positions (integer crossings of `U`/`V`), under one-to-one matching |

`grid_align` is the metric that no prior tool offered: it uses the engine's
exact per-pixel source-coordinate maps as ground truth for **where the grid
lines actually are**, independent of colour and even of getting the cell count
exactly right. Matching is **one-to-one**, so a method that sprays extra cut
lines cannot inflate its score - each matched line consumes a distinct true cut.

## palette

Did the method recover the right set of colours?

| Field | Better | Meaning |
|---|---|---|
| `ncolor_err` | lower | relative error in the number of distinct colours vs the source palette |
| `palette_de` | lower | mean CIELAB distance from each recovered colour to its nearest source colour |

## Aggregation

Rate fields (`exact`, `within1`, `pixel_match`, `grid_align`, ...) are averaged;
error fields (`rel_err`, `delta_e`, `psnr`, `palette_de`, `sec`) use the median,
which is robust to the occasional catastrophic outlier. This is done per
distortion category and overall.

## Adding a metric

Drop a file in `pixelbench/metrics/` with a `@register` class exposing `name`,
`fields` (field -> `"higher"|"lower"`), and `score(sample, rec) -> {field: value}`.
Return `{}` to skip (for example when a detect-only method returns no native
image). See [../CONTRIBUTING.md](../CONTRIBUTING.md).
