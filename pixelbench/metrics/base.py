"""Metric plugin API.

A *metric* turns one (ground-truth sample, method reconstruction) pair into
one or more named scalar signals. Adding one is a single file:

    # pixelbench/metrics/mymetric.py
    from .base import Metric, register

    @register
    class MyMetric(Metric):
        name = "mymetric"
        fields = {"my_score": "higher"}     # field -> "higher"|"lower" is better
        def score(self, sample, rec):
            return {"my_score": ...}         # return {} to skip (e.g. no native)

`sample` carries the clean ground truth and the exact distortion geometry:

    sample.gt        HxWx3 uint8   the clean 1x source (intended output)
    sample.distorted HxWx4 uint8   the input the method was given
    sample.U, sample.V  float32     per-output-pixel source coords (grid truth)
    sample.gt_cols, sample.gt_rows  true native size
    sample.category, sample.severity, sample.image_id
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

import numpy as np

from ..methods.base import Reconstruction


@dataclass
class Sample:
    gt: np.ndarray
    distorted: np.ndarray
    U: np.ndarray
    V: np.ndarray
    gt_cols: int
    gt_rows: int
    category: str
    severity: float
    image_id: str


class Metric:
    name: str = ""
    # field name -> "higher" or "lower" (which direction is better)
    fields: dict[str, str] = {}

    def score(self, sample: Sample, rec: Reconstruction) -> dict[str, float]:
        raise NotImplementedError


_REGISTRY: dict[str, type[Metric]] = {}


def register(cls: type[Metric]) -> type[Metric]:
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    if cls.name in _REGISTRY:
        raise ValueError(f"duplicate metric name {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def _discover() -> None:
    import pixelbench.metrics as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name != "base":
            importlib.import_module(f"pixelbench.metrics.{m.name}")


def all_metrics() -> list[Metric]:
    _discover()
    return [_REGISTRY[n]() for n in sorted(_REGISTRY)]


def field_directions() -> dict[str, str]:
    """Flat map of every metric field -> 'higher'|'lower'."""
    out = {}
    for m in all_metrics():
        out.update(m.fields)
    return out
