# Known-scale reconstruction-quality protocol

## Purpose

This track compares output quality when every method is given the same correct
native dimensions. It is suitable for scale-specific models and production
pipelines where output size is an explicit user or system input.

It is complementary to the original PixelBench automatic-grid track. It must
not be described as the official PixelBench leaderboard protocol.

## Required controls

1. Start from true native 1× pixel art with documented provenance and licences.
2. Generate distorted inputs using a frozen PixelBench suite version and seed.
3. Supply the target native width and height equally to every method that can
   accept them. Record when a method cannot accept this information.
4. Preserve the complete output of each method before scoring.
5. Score against the same clean native target and the same metric version.
6. Record failures and skips; do not silently remove them.

For the initial known-scale release, the `fractional` and `nonsquare` categories
are excluded from the primary 41-category comparison. Their central challenge
is scale or per-axis grid inference, which this track supplies by design. They
remain appropriate—and important—in the original 43-category automatic-grid
track.

## What may rank methods

The six primary expanded metrics may be compared independently:

- `delta_e00_mean`
- `delta_e00_p95`
- `edge_assd_px`
- `edge_palette_violation_rate`
- `palette_emd_de00`
- `palette_chamfer_de00`

Original PixelBench colour, palette, and pixel-reconstruction fields should
also remain in a compatibility table. In a known-scale run, resolution fields
and `grid_align` primarily confirm that the runner honoured the supplied output
contract; they must not be used to imply superior grid inference.

## Aggregation and uncertainty

The source image is the statistical unit. Multiple distortions of one image are
correlated repeated measurements, not additional independent artworks.

Report:

- the number of unique source images;
- the number of generated cases and cases successfully scored;
- per-scale and overall values;
- per-category values where practical;
- a declared aggregate (mean or median) for each metric; and
- 95% confidence intervals produced by resampling source-image IDs, keeping all
  of an image's distortions together in each bootstrap draw.

Do not change aggregation based on which method looks best. Do not replace the
six metrics with an undocumented composite ranking.

## Fairness disclosures

Every publication must state:

- whether dimensions were detected, selected by a user, or supplied from the
  target metadata;
- whether each method ran locally, through an API, or through a website;
- the exact method version, model, settings, and date of external execution;
- whether any method had access to training-related or proprietary evaluation
  images;
- all preprocessing and postprocessing, including tiling; and
- corpus ownership, redistribution rights, and any private-data limitation.

Results on private data can be useful internal evidence but are not independently
reproducible. A public release should include a redistributable corpus manifest
and hashes, or a deterministic acquisition procedure where redistribution is
not permitted.
