# Upstream provenance

This staging repository was created from a clean archive of:

- Project: PixelBench
- Repository: <https://github.com/Retro-Diffusion/pixel-bench>
- Commit: `779cfc2548ac0da5957514e76d77d8f878ecb513`
- Licence: MIT
- Upstream copyright: Copyright (c) 2026 Astropulse, LLC

The upstream `LICENSE` file is preserved without modification.

## Material changes in this staging fork

- Added `pixelbench.metrics.expanded`, a registered complementary metric plugin.
- Added controlled behavioural tests for the expanded metrics.
- Added a deterministic metric-validation script and checked-in validation
  artefacts.
- Added scikit-image as a dependency for CIEDE2000 and SSIM.
- Added documentation separating the original automatic-grid benchmark from
  the known-scale reconstruction-quality protocol.
- Added public-release hygiene rules for credentials, model weights, generated
  data, and external-service caches.

The original PixelBench metrics are intentionally retained with their exact
field names. Fork results should be identified as PixelBench Expanded results,
not official PixelBench leaderboard submissions.

This fork is independently maintained and is not presented as affiliated with
or endorsed by Astropulse, Retro Diffusion, or the upstream authors.
