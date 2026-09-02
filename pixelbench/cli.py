"""pixel-bench command line.

    pixelbench run --data ./my_pixel_art [--methods fixer,snapper] [--limit N]
    pixelbench report results/run.json [--out REPORT.md]
    pixelbench list-methods | list-metrics | list-suite
    pixelbench validate --data ./my_pixel_art
"""
from __future__ import annotations

import argparse
import os
import sys


def _cmd_run(args):
    from .runner import run
    methods = args.methods.split(",") if args.methods else None
    categories = args.categories.split(",") if args.categories else None
    run(args.data, args.out, methods=methods, categories=categories,
        severity=args.severity, limit=args.limit, workers=args.workers,
        seed=args.seed)
    if not args.no_report:
        from .report import render_file
        rep = os.path.splitext(args.out)[0] + ".md"
        render_file(args.out, rep)
        print(f"report -> {rep}")


def _cmd_report(args):
    from .report import render_file
    out = args.out or (os.path.splitext(args.results)[0] + ".md")
    render_file(args.results, out)
    print(f"report -> {out}")


def _cmd_list_methods(args):
    from .methods import registry_status
    print("methods (name / available / requires):")
    for name, avail, req in registry_status():
        tag = "" if avail else "  [missing: %s]" % ",".join(req)
        print(f"  {name:16s} {'yes' if avail else 'no ':3s}{tag}")


def _cmd_list_metrics(args):
    from .metrics import all_metrics
    print("metrics (name -> fields):")
    for m in all_metrics():
        fields = ", ".join(f"{k} ({v})" for k, v in m.fields.items())
        print(f"  {m.name:14s} {fields}")


def _cmd_list_suite(args):
    from .distort.suite import CATEGORIES, SUITE_VERSION
    print(f"distortion suite {SUITE_VERSION}:")
    for c in CATEGORIES:
        print(f"  {c.name:14s} {c.description}")


def _cmd_preview(args):
    from .preview import make_preview, preview_dataset
    if args.image:
        out = args.out or "preview.png"
        make_preview(args.image, out, tile=args.tile, severity=args.severity)
        print(f"preview -> {out}")
    else:
        outs = preview_dataset(args.data, args.out or "previews", n=args.n,
                               severity=args.severity, tile=args.tile)
        for o in outs:
            print(f"preview -> {o}")


def _cmd_validate(args):
    from .data import list_images, looks_upscaled
    paths = list_images(args.data)
    print(f"{len(paths)} candidate images in {args.data}")
    flagged = 0
    for p in paths:
        bad, why = looks_upscaled(p)
        if bad:
            flagged += 1
            if flagged <= 20:
                print(f"  WARN {os.path.basename(p)}: {why}")
    if flagged:
        print(f"\n{flagged}/{len(paths)} images look pre-scaled (not native "
              f"1x). pixel-bench expects true 1x pixel art - scores on scaled "
              f"inputs are meaningless.")
    else:
        print("all images look like native 1x pixel art.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pixelbench",
                                description="Open benchmark for pixel-art reconstruction.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the benchmark")
    r.add_argument("--data", required=True, help="folder of native 1x pixel art")
    r.add_argument("--out", default="results/run.json")
    r.add_argument("--methods", help="comma list (default: all available)")
    r.add_argument("--categories", help="comma list (default: whole suite)")
    r.add_argument("--severity", type=float, default=1.0)
    r.add_argument("--limit", type=int, help="sample this many source images")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--no-report", action="store_true")
    r.set_defaults(func=_cmd_run)

    rp = sub.add_parser("report", help="render a report from results JSON")
    rp.add_argument("results")
    rp.add_argument("--out")
    rp.set_defaults(func=_cmd_report)

    sub.add_parser("list-methods").set_defaults(func=_cmd_list_methods)
    sub.add_parser("list-metrics").set_defaults(func=_cmd_list_metrics)
    sub.add_parser("list-suite").set_defaults(func=_cmd_list_suite)

    v = sub.add_parser("validate", help="check a dataset is native 1x")
    v.add_argument("--data", required=True)
    v.set_defaults(func=_cmd_validate)

    pv = sub.add_parser("preview", help="render a montage of the distortion suite")
    pv.add_argument("--data", help="folder of 1x art (previews the first --n)")
    pv.add_argument("--image", help="a single image to preview")
    pv.add_argument("--out", help="output png (single) or folder (dataset)")
    pv.add_argument("--n", type=int, default=1)
    pv.add_argument("--tile", type=int, default=200, help="tile size in px")
    pv.add_argument("--severity", type=float, default=1.0)
    pv.set_defaults(func=_cmd_preview)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
