# Metric behaviour validation

Controlled perturbations isolate one failure mode at a time. Passing means the metric responds monotonically in the documented direction; it does not establish universal perceptual validity.

| Metric | Perturbation | Values | Result |
|---|---|---|---|
| `delta_e00_mean` | Red-channel drift | 0.0000, 1.2513, 2.4264, 3.4945, 4.4721 | PASS |
| `delta_e00_p95` | Red-channel drift | 0.0000, 5.3902, 10.1639, 14.1033, 17.3305 | PASS |
| `edge_assd_px` | Translation (native pixels) | 0.0000, 0.1760, 0.3666, 0.4681 | PASS |
| `edge_palette_violation_rate` | Blended edge coverage (%) | 0.0000, 1.0417, 2.0833, 3.1250, 4.1667, 5.2083 | PASS |
| `palette_emd_de00` | Deleted palette colours | 0.0000, 18.2981, 27.6706, 33.3877 | PASS |
| `palette_chamfer_de00` | Deleted palette colours | 0.0000, 9.1491, 11.8683, 16.6938 | PASS |
