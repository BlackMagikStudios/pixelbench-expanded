# Expanded metrics

This document defines the complementary full-reference metrics added by
PixelBench Expanded. Every score compares a reconstructed native RGB image with
the clean native RGB target. If dimensions differ, the reconstruction is first
resized to the target dimensions with nearest-neighbour sampling. Resolution
must still be reported separately because this alignment can hide a wrong size.

No aggregate “winner score” is defined. Different metrics measure different
failure modes and must remain visible.

## Primary metrics

### `delta_e00_mean` and `delta_e00_p95`

Each sRGB image is converted to CIELAB using `skimage.color.rgb2lab`. Per-pixel
colour differences use CIEDE2000 (`deltaE_ciede2000`). The mean summarizes
overall colour fidelity; the 95th percentile exposes severe local errors that
can be diluted in an average.

- Better: lower
- Unit: CIEDE2000 colour difference
- Identity: 0
- Limitation: spatial misplacement and colour error are entangled. A shifted
  but otherwise correct region can have high colour error, which is why edge
  displacement is also reported.

### `edge_assd_px`

An edge pixel is any target or candidate pixel that participates in a
4-connected horizontal or vertical colour transition greater than ΔE00 2.0.
For both edge masks, an Euclidean distance transform gives the distance from
each edge pixel to the nearest edge in the other image. `edge_assd_px` is the
mean of the two directed sets of distances.

- Better: lower
- Unit: native pixels
- Identity: 0
- Empty-mask rule: two empty masks score 0; a one-sided empty mask receives the
  image-diagonal penalty.
- Limitation: it measures where colour boundaries occur, not whether the two
  regions have the correct semantic colour. Dense texture can also create many
  boundaries.

Average symmetric surface distance is an established boundary-distance
construction. Our definition of a pixel-art colour boundary and its ΔE00
threshold are benchmark-specific and are therefore frozen and tested here.

### `edge_palette_violation_rate`

Target edges are detected as above and dilated by a one-pixel square radius to
form an evaluation band. For each pixel in this band, the candidate colour is
compared with every target colour in the corresponding 3×3 neighbourhood. It is
a violation when the nearest local target colour is more than ΔE00 2.0 away.
The metric is the fraction of band pixels that violate this rule.

- Better: lower
- Range: 0–1
- Identity: 0
- Intended sensitivity: blended edge colours, halos, and invented edge colours
  that do not belong to either nearby target region.
- Intended tolerance: a legitimate colour displaced by one native pixel can
  match the local neighbourhood and is not automatically called a mixel.
- Limitation: this is a new, domain-specific metric—not an external standard.
  The local window can forgive errors near very dense multicolour detail, and
  the fixed threshold may not match every display or artistic intent.

### `palette_emd_de00`

Each image becomes a distribution over RGB palette entries. The weight of an
entry is its fraction of pixels; the transport cost between two entries is
CIEDE2000. OpenCV's Earth Mover's Distance solves the resulting transportation
problem. Palettes above 256 entries are reduced by median-cut quantization with
dithering disabled before measurement.

- Better: lower
- Unit: transported ΔE00 cost
- Identity: 0
- Strength: considers both colour distance and how much of the image uses each
  colour.
- Limitation: it discards spatial arrangement. Two images can share the same
  palette distribution while placing colours differently. The 256-colour cap
  is a declared engineering bound and should be included in protocol versions.

Earth Mover's Distance is established; using ΔE00 palette entries and pixel
frequency as its ground distance and mass is this benchmark's application of
the method.

### `palette_chamfer_de00`

For every target palette colour, find the nearest candidate palette colour in
ΔE00; repeat from candidate to target; then average the two means. Palette
entries are unweighted here, so a rare accent colour can still affect the
score. The same 256-colour cap applies.

- Better: lower
- Unit: CIEDE2000 colour difference
- Identity: 0
- Strength: symmetric direction detects both missing target colours and
  colours invented by the method.
- Limitation: because entries are unweighted, a one-pixel colour and a dominant
  background colour contribute equally. It complements rather than replaces
  the frequency-aware EMD.

Symmetric nearest-neighbour distance is a Chamfer-style construction used for
set comparison. Applying it to palette support in ΔE00 space is our adaptation.

## Diagnostic fields

The implementation also exposes these investigation aids. They are not part of
the primary six-metric claim:

| Field | Better | Purpose |
|---|---:|---|
| `foreground_delta_e00_mean` | lower | Mean ΔE00 over a supplied foreground mask; defaults to the full image when none is supplied. |
| `local_delta_e00_mean` | lower | Colour error allowing a 3×3 local match; helps separate small displacement from invented colour. |
| `color_match_de00_2` | higher | Share of pixels within ΔE00 2.0. |
| `ssim` | higher | Structural similarity diagnostic. |
| `pixel_match_tol4` | higher | Share of pixels with maximum channel error no greater than 4. |
| `edge_exact_f1` | higher | Exact overlap F1 for the colour-edge masks. |
| `edge_tolerant_f1` | higher | Edge-mask F1 with one-pixel tolerance. |
| `edge_hd95_px` | lower | Robust 95th-percentile symmetric edge distance. |
| `palette_size_log_error` | lower | Absolute log2 ratio between candidate and target palette sizes, with +1 stabilization. |
| `native_size_exact` | higher | Internal full-reference size contract check. |

PixelBench's original `psnr` remains owned by the unchanged `color` metric and
is not registered a second time by the extension.

## Fixed parameters

The initial metric protocol freezes:

- sRGB-to-CIELAB conversion: scikit-image;
- perceptual difference: CIEDE2000;
- edge transition threshold: ΔE00 > 2.0;
- edge evaluation dilation: one pixel with a 3×3 square structuring element;
- local palette window: 3×3;
- local violation threshold: nearest target colour ΔE00 > 2.0;
- palette cap: 256 entries;
- cap method: Pillow median cut, no dithering; and
- spatial distance unit: native image pixels.

Changing any of these creates a new protocol version and must not be merged
silently with older results.

## Validation scope

`tests/test_expanded_metrics.py` contains controlled behavioural checks and
`tools/validate_expanded_metrics.py` generates monotonic perturbation curves.
These tests answer “does the implementation respond in the intended direction
to known damage?” They do not establish perceptual superiority or remove the
need for diverse art, external implementations, confidence intervals, and
future correlation studies.
