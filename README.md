# PixelBench Expanded

PixelBench Expanded is a dataset-independent benchmark package. It contains the
benchmark engine, metric definitions, validation tests, reporting tools, and
protocol documentation. It intentionally contains no evaluation corpus, model
weights, external-service cache, or product benchmark results.

PixelBench Expanded is a research-oriented fork of
[PixelBench](https://github.com/Retro-Diffusion/pixel-bench), an open benchmark
for reconstructing native pixel art from enlarged, blurred, compressed,
misaligned, or otherwise damaged images.

PixelBench is an unusually strong foundation for this problem. It provides a
deterministic, versioned distortion suite; keeps the clean native image as
ground truth; separates resolution, colour, placement, grid, and palette
signals; and exposes a modular interface for methods and metrics. Those choices
make comparisons reproducible and make failures easier to diagnose than a
single opaque quality score would.

This fork preserves those original metrics and adds a complementary quality
track. We made the extension because some production pixel-art systems are
given the requested native dimensions. In that setting, resolution recovery
and grid detection are controlled inputs rather than the main research
question. Two outputs can therefore receive the same perfect size score while
looking very different: one may retain clean, correctly placed palette colours,
while another contains shifted outlines, blended edge pixels, missing accent
colours, or a few severe colour failures hidden by an average.

The extension measures those remaining differences. It does **not** claim that
PixelBench is incorrect, and its numbers must not be presented as entries on
the official PixelBench leaderboard.

## Two tracks, two questions

| Track | Question | Dimensions |
|---|---|---|
| **Original PixelBench / automatic grid** | Can the method infer the native grid and reconstruct the art? | Unknown to the method; all 43 upstream distortion categories are valid. |
| **Known-scale reconstruction quality** | Given the same correct native dimensions, which method produces the most faithful native pixel art? | Supplied equally to every method; size and grid metrics are reported as contract checks, not ranking evidence. |

The tracks should be reported separately. Mixing them would reward different
capabilities in the same table and make the result difficult to interpret. See
[the known-scale protocol](docs/KNOWN_SCALE_PROTOCOL.md).

## What remains unchanged

The fork retains PixelBench's original metric names and implementations:

- `exact`, `within1`, `within5pct`, and `rel_err` for native resolution;
- `pixel_match` and `grid_align` for pixel placement and grid recovery;
- `delta_e` and `psnr` for colour and signal fidelity; and
- `ncolor_err` and `palette_de` for palette recovery.

These form the compatibility section of a report. The upstream code lives in
the same module structure so its formulas remain auditable.

## What the expanded track adds

The primary added metrics are deliberately reported separately—there is no
custom composite score and no hidden weighting:

| Metric | Better | What it reveals |
|---|---:|---|
| `delta_e00_mean` | lower | Mean perceptual colour difference using CIEDE2000. |
| `delta_e00_p95` | lower | Severe colour failures that a mean can hide. |
| `edge_assd_px` | lower | Average symmetric displacement of colour boundaries, in native pixels. |
| `edge_palette_violation_rate` | lower | Edge pixels whose colour is not supported by the nearby target palette; aimed at blended “mixels” and invented edge colours. |
| `palette_emd_de00` | lower | Frequency-aware transport cost between target and output palettes. |
| `palette_chamfer_de00` | lower | Symmetric palette-support distance, sensitive to both missing and invented colours. |

Additional diagnostic fields are available for investigation, but are not part
of the primary six-metric comparison. Exact formulas, frozen thresholds,
failure modes, and the distinction between established constructions and our
pixel-art-specific adaptation are documented in
[Expanded Metrics](docs/EXPANDED_METRICS.md).

## Why these metrics are defensible

The suite combines established ideas rather than inventing a score that happens
to favour one model:

- CIEDE2000 is a standardized perceptual colour-difference formula with a
  published implementation analysis by Sharma, Wu, and Dalal.
- Earth Mover's Distance compares distributions while accounting for the cost
  of moving probability mass; here the mass is colour frequency and the cost is
  CIEDE2000 colour distance.
- Symmetric nearest-neighbour (Chamfer-style) distance checks palette support in
  both directions, so missing and invented colours are both visible.
- Symmetric surface distance is a conventional way to measure displacement
  between boundaries rather than only their overlap.
- The edge-palette violation rate is our domain-specific metric. It is included
  openly, with fixed parameters, controlled perturbation tests, and explicit
  limitations—not presented as an external standard.

This follows the broader recommendation in *Metrics Reloaded*: choose metrics
for the property and failure mode being evaluated, use complementary metric
families, and document their pitfalls. That paper concerns biomedical image
analysis; we use its metric-selection principles, not its application-specific
claims. Full citations and the precise scope of each source are in
[References](docs/REFERENCES.md).

## Validation

Metric behaviour is tested against controlled pixel-art perturbations. The
tests check, among other things, that identity is ideal, increasing colour drift
raises colour error, increasing translation raises boundary distance, invented
or deleted colours raise palette distances, and a valid one-pixel colour move
is not mislabeled as a new edge colour.

```bash
python -m pip install -e ".[report]"
python -m unittest discover -s tests -v
python tools/validate_expanded_metrics.py
```

The checked-in validation artefacts are under [`validation/`](validation/).
They validate expected behaviour on controlled cases; they are not a substitute
for external validation or a diverse public evaluation corpus.

## Running the original benchmark

The original PixelBench command-line flow is preserved:

```bash
pixelbench validate --data ./native_pixel_art
pixelbench run --data ./native_pixel_art --out results/run.json
pixelbench report results/run.json
```

Native source images must be true 1× pixel art: one stored pixel must represent
one art pixel. See the upstream-style [distortion documentation](docs/DISTORTIONS.md)
and [results format](docs/RESULTS_FORMAT.md).

## Reporting requirements

A publishable comparison should disclose:

- the benchmark and protocol version;
- corpus source, licence, image count, and content-selection rules;
- the number of source images as the statistical unit, not only the number of
  generated distortions;
- the method version and every option, including supplied dimensions;
- per-metric direction and units;
- aggregate values plus per-scale results and uncertainty intervals; and
- failures, skips, exclusions, and any unavailable method.

Do not describe repeated distortions of the same source art as independent
images. Do not combine official PixelBench results with known-scale fork results
without a prominent protocol distinction.

## Evaluation data

No evaluation images are distributed with this repository. Users provide their
own native 1× pixel-art corpus; PixelBench Expanded deterministically creates
paired damaged inputs while retaining the clean native images as references.

Black Magik's internal evaluation used a held-out corpus of 100 source images
and 4,100 paired cases. That corpus and its product results are not included in
this repository. Those internal numbers are not independently reproducible from
this package alone; the benchmark methodology is reproducible with a
user-supplied corpus.

## Project status and attribution

This fork was derived from PixelBench commit
`779cfc2548ac0da5957514e76d77d8f878ecb513`. PixelBench is copyright
Astropulse, LLC and distributed under the MIT License. The original
[LICENSE](LICENSE) is preserved. This fork is independently maintained and is
not presented as an official PixelBench release or as endorsed by its authors.

See [UPSTREAM.md](UPSTREAM.md) for provenance.
