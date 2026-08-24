"""CLI for a caller-prepared, portable accepted-campaign evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .conditions import CONDITIONS, DATASETS, resolve_condition
from .model_registry import MODELS
from .runner import run_prepared_evaluation


def main(argv: list[str] | None = None) -> int:
    """Execute one fail-closed public Qwen campaign profile."""
    parser = argparse.ArgumentParser(prog="evaluation-footprint")
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--prepared-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--subset-rows", type=int)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--max-pixels-override", type=int)
    parser.add_argument("--telemetry", action="store_true", help="Measure prediction-generation footprint only.")
    parser.add_argument("--telemetry-device-index", type=int)
    parser.add_argument("--carbon-profile", choices=("accepted-campaign",))
    parser.add_argument("--carbon-intensity-g-per-kwh", type=float)
    args = parser.parse_args(argv)
    if not args.prepared_dataset.exists():
        parser.error("prepared dataset path does not exist")
    if args.membership is not None and not args.membership.is_file():
        parser.error("membership path does not exist")
    if args.carbon_profile is not None and args.carbon_intensity_g_per_kwh is not None:
        parser.error("choose either --carbon-profile or --carbon-intensity-g-per-kwh")
    if (args.carbon_profile is not None or args.carbon_intensity_g_per_kwh is not None) and not args.telemetry:
        parser.error("carbon options require --telemetry")
    condition = resolve_condition(
        args.condition,
        model_key=args.model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        subset_rows=args.subset_rows,
        replicate=args.replicate,
        max_pixels_override=args.max_pixels_override,
    )
    run_prepared_evaluation(
        model_key=args.model,
        dataset=args.dataset,
        condition=condition,
        prepared_dataset=args.prepared_dataset,
        output=args.output,
        membership_path=args.membership,
        telemetry=args.telemetry,
        telemetry_device_index=args.telemetry_device_index,
        carbon_profile=args.carbon_profile,
        carbon_intensity_g_per_kwh=args.carbon_intensity_g_per_kwh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
