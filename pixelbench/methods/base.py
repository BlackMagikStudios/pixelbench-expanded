"""Reconstruction-method plugin API.

A *method* takes a distorted image and returns its best guess at the native
(1x) pixel art. Adding one is a single file:

    # pixelbench/methods/mytool.py
    from .base import Method, Reconstruction, register

    @register
    class MyTool(Method):
        name = "mytool"
        requires = ("mytool_pkg",)          # optional deps; missing -> skipped
        def reconstruct(self, image):        # image: HxWx4 uint8 (distorted)
            native = ...                      # recovered 1x art, uint8 HxWx3/4
            return Reconstruction(native=native)   # cols/rows inferred if omitted

Drop the file in this folder and it self-registers - the runner and CLI pick
it up with no other edits.
"""
from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass

import numpy as np


@dataclass
class Reconstruction:
    """A method's output. `native` is the recovered 1x art (uint8, HxWx3 or
    HxWx4). `cols`/`rows` are the predicted native width/height; if omitted
    they are inferred from `native.shape`. A detect-only method may return
    cols/rows with native=None (color/placement metrics are then skipped)."""
    native: np.ndarray | None = None
    cols: int | None = None
    rows: int | None = None

    def size(self) -> tuple[int | None, int | None]:
        if self.cols and self.rows:
            return int(self.cols), int(self.rows)
        if self.native is not None:
            return int(self.native.shape[1]), int(self.native.shape[0])
        return None, None


class Method:
    """Base class for reconstruction methods. Subclass, set `name`, implement
    `reconstruct`. Declare third-party imports in `requires` so the method is
    skipped cleanly when they are not installed."""
    name: str = ""
    requires: tuple[str, ...] = ()

    def available(self) -> bool:
        for mod in self.requires:
            if importlib.util.find_spec(mod) is None:
                return False
        return True

    def reconstruct(self, image: np.ndarray) -> Reconstruction:
        raise NotImplementedError


_REGISTRY: dict[str, type[Method]] = {}


def register(cls: type[Method]) -> type[Method]:
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    if cls.name in _REGISTRY:
        raise ValueError(f"duplicate method name {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def _discover() -> None:
    """Import every sibling module so @register decorators run."""
    import pixelbench.methods as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name != "base":
            importlib.import_module(f"pixelbench.methods.{m.name}")


def all_methods(only_available: bool = True) -> list[Method]:
    _discover()
    out = []
    for name in sorted(_REGISTRY):
        inst = _REGISTRY[name]()
        if only_available and not inst.available():
            continue
        out.append(inst)
    return out


def get_method(name: str) -> Method:
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def registry_status() -> list[tuple[str, bool, tuple[str, ...]]]:
    """(name, available, requires) for every registered method."""
    _discover()
    rows = []
    for name in sorted(_REGISTRY):
        inst = _REGISTRY[name]()
        rows.append((name, inst.available(), inst.requires))
    return rows
