# References and scope

These sources support the individual mathematical constructions or the metric-
selection methodology. None independently validates the complete PixelBench
Expanded suite for pixel-art reconstruction. The pixel-art application,
thresholds, and protocol still require transparent validation and community
review.

## Upstream benchmark

**Astropulse, LLC.** “PixelBench: an open benchmark for pixel-art
reconstruction.” GitHub repository, 2026.
[Repository](https://github.com/Retro-Diffusion/pixel-bench)

Supports the distortion engine, original metrics, adapter structure, and result
format inherited by this fork.

## Perceptual colour difference

**Sharma, G., Wu, W., and Dalal, E. N.** “The CIEDE2000 Color-Difference
Formula: Implementation Notes, Supplementary Test Data, and Mathematical
Observations.” *Color Research & Application* 30(1), 21–30 (2005).
[DOI](https://doi.org/10.1002/col.20070) ·
[Author manuscript](https://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf)

Supports the definition and implementation checking of CIEDE2000. It does not
establish that any single ΔE00 threshold is universal for pixel art.

## Distribution distance

**Rubner, Y., Tomasi, C., and Guibas, L. J.** “The Earth Mover's Distance as a
Metric for Image Retrieval.” *International Journal of Computer Vision* 40,
99–121 (2000).
[DOI](https://doi.org/10.1023/A:1026543900054) ·
[Paper](https://ai.stanford.edu/~rubner/papers/rubnerIjcv00.pdf)

Supports Earth Mover's Distance for comparing weighted distributions under a
ground distance. Treating pixel frequency as mass and ΔE00 as palette ground
cost is this fork's application.

## Set distance

**Fan, H., Su, H., and Guibas, L. J.** “A Point Set Generation Network for 3D
Object Reconstruction from a Single Image.” *CVPR* (2017).
[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fan_A_Point_Set_CVPR_2017_paper.pdf)

Provides a modern, explicit use of Chamfer and Earth Mover distances for
unordered set comparison. It supports the symmetric nearest-neighbour
construction, not the claim that palette Chamfer alone measures pixel-art
quality.

## Boundary evaluation

**Cheng, B., Girshick, R., Dollár, P., Berg, A. C., and Kirillov, A.**
“Boundary IoU: Improving Object-Centric Image Segmentation Evaluation.” *CVPR*
(2021).
[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)

Supports the general need for boundary-sensitive evaluation because region-
dominated metrics can obscure boundary quality. This fork uses surface
distance, not Boundary IoU, and defines edges from colour transitions rather
than segmentation labels.

**Martin, D., Fowlkes, C., Tal, D., and Malik, J.** “A Database of Human
Segmented Natural Images and its Application to Evaluating Segmentation
Algorithms and Measuring Ecological Statistics.” *ICCV* (2001), with the
Berkeley Segmentation Dataset and Benchmark.
[Project](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)

Provides established context for evaluating boundaries with localization
tolerance. It does not validate this fork's colour-edge detector.

## Metric selection and limitations

**Maier-Hein, L. et al.** “Metrics Reloaded: recommendations for image analysis
validation.” *Nature Methods* 21, 195–212 (2024).
[DOI](https://doi.org/10.1038/s41592-023-02151-z) ·
[Open-access article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/)

Supports problem-aware metric selection, complementary metric families, and
explicit attention to metric pitfalls. Its original domain is biomedical image
analysis; this fork applies the selection principles rather than transferring
domain-specific validation claims.

## Diagnostic structural similarity

**Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P.** “Image Quality
Assessment: From Error Visibility to Structural Similarity.” *IEEE Transactions
on Image Processing* 13(4), 600–612 (2004).
[DOI](https://doi.org/10.1109/TIP.2003.819861) ·
[IEEE](https://ieeexplore.ieee.org/document/1284395)

Supports SSIM, included only as a diagnostic. SSIM was designed for general
image quality and may reward smooth structural similarity that conflicts with
strict pixel-art requirements; it is therefore not a primary metric here.
