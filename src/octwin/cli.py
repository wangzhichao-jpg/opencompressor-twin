"""One local CLI for both structures and all manufacturers."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .core import TARGETS, load_dataset, load_points, read_rows
from .properties import suction_properties
from .reporting import report, write_csv
from .workflow import benchmark, build_artifact, load_artifact, predict_point, recommend, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="octwin", description="Unified compressor engineering preview; no real-data validation claim by default"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "calibrate", "benchmark", "learning-curve", "adaptive-replay", "development-study"):
        p = commands.add_parser(name)
        p.add_argument("data", help="Observed SI CSV")
        p.add_argument("--meta", required=True, help="JSON metadata")
        if name != "validate":
            p.add_argument("--output", required=True, help="Model JSON (calibrate) or run directory")
            p.add_argument(
                "--split",
                required=name in {"learning-curve", "adaptive-replay", "development-study"},
                help="Frozen JSON group split/development protocol",
            )
        if name == "calibrate":
            p.add_argument(
                "--residual-rbf",
                action="store_true",
                help="Explicit research RBF log-residual extension; no new validation claim",
            )
        if name == "benchmark":
            p.add_argument("--seed", type=int, default=42)
        if name in {"learning-curve", "adaptive-replay"}:
            p.add_argument("--budgets", default="9,12,16,24,32,40", help="Predeclared comma-separated group budgets")
            p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", help="Fixed repeated ordering seeds")
    for name in ("predict", "recommend"):
        p = commands.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--conditions" if name == "predict" else "--candidates", required=True)
        p.add_argument("--output", required=True)
        if name == "recommend":
            p.add_argument("--budget", type=int, default=1)
            p.add_argument("--strategy", choices=("coverage", "d_optimal", "kernel_ivr"), default="coverage")
            p.add_argument(
                "--cost-column",
                help="Optional positive relative-cost input column for d_optimal/kernel_ivr; budget counts points, not total time",
            )
            p.add_argument(
                "--approved",
                action="store_true",
                required=True,
                help="Confirm candidate inputs have been approved by an engineer; not approval by this software",
            )
    p = commands.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--output")
    p = commands.add_parser("demo")
    p.add_argument("--output", default="runs/software_demo")
    args = parser.parse_args(argv)
    try:
        if args.command in {
            "validate",
            "calibrate",
            "benchmark",
            "learning-curve",
            "adaptive-replay",
            "development-study",
        }:
            if (
                args.command == "development-study"
                and json.loads(Path(args.meta).read_text(encoding="utf-8")).get("research_partition")
                != "development_only"
            ):
                raise ValueError(
                    "Development study requires separate development-only metadata/CSV, not original holdout data"
                )
            dataset = load_dataset(args.data, args.meta)
            for o in dataset.observations:
                suction_properties(o.point, dataset.spec.refrigerant)
            if args.command == "validate":
                answer = {
                    "status": "schema_and_single_phase_checks_passed",
                    "compressor": asdict(dataset.spec),
                    "rows": len(dataset.observations),
                    "groups": len({o.group_id for o in dataset.observations}),
                    "fingerprint": dataset.fingerprint,
                    "source_kind": dataset.metadata["source_kind"],
                    "speed_definition": dataset.metadata["speed_definition"],
                    "nominal_speed_provenance": dataset.metadata.get("nominal_speed_provenance"),
                    "warning": "Schema/physics checks do not admit a source as independent experimental validation",
                }
            else:
                split = json.loads(Path(args.split).read_text(encoding="utf-8")) if args.split else None
                if args.command == "benchmark":
                    answer = benchmark(args.data, args.meta, args.output, split, args.seed)
                    answer = {
                        "evidence_label": answer["evidence_label"],
                        "metrics": answer["metrics"],
                        "report": str(Path(args.output) / "report.html"),
                    }
                elif args.command == "development-study":
                    from .development import run_development_study

                    value = run_development_study(args.data, args.meta, split, args.output)
                    answer = {
                        "report": str(Path(args.output) / "report.html"),
                        "records": len(value["rows"]),
                        "unavailable_or_failed_trials": len(value["failures"]),
                        "audit": value["audit"],
                        "note": "Development-only CV; original calibration/final labels excluded; NOT new blind validation",
                    }
                elif args.command in {"learning-curve", "adaptive-replay"}:
                    if args.command == "learning-curve":
                        from .evaluation import run_learning_curve as runner
                    else:
                        from .replay import run_adaptive_replay as runner
                    budgets = tuple(int(v.strip()) for v in args.budgets.split(",") if v.strip())
                    seeds = tuple(int(v.strip()) for v in args.seeds.split(",") if v.strip())
                    value = runner(args.data, args.meta, split, args.output, budgets, seeds)
                    answer = {
                        "report": str(Path(args.output) / "report.html"),
                        "trials": len(value["rows"]),
                        "unavailable_or_failed_trials": len(value["failures"]),
                        "protocol_sha256": value["protocol_sha256"],
                        "selection_audit": value.get("selection_audit"),
                        "note": "Secondary retrospective fixed-budget analysis, not new blind validation or measured-time savings",
                    }
                else:
                    if split is not None:
                        from .workflow import check_split

                        check_split(dataset.observations, split)
                        observations = [o for o in dataset.observations if o.group_id in split["train"]]
                    else:
                        observations = dataset.observations
                    artifact = build_artifact(dataset, observations)
                    if args.residual_rbf:
                        from .discrepancy import add_residual_correction

                        artifact = add_residual_correction(artifact, observations)
                    write_json(args.output, artifact)
                    answer = {
                        "saved": args.output,
                        "diagnostics": artifact["diagnostics"],
                        "note": "Calibration only; not a holdout score",
                    }
        elif args.command in {"predict", "recommend"}:
            artifact = load_artifact(args.model)
            file = args.conditions if args.command == "predict" else args.candidates
            if args.command == "recommend" and any(k in read_rows(file)[0] for k in TARGETS):
                raise ValueError("Recommendation requires an input-only candidate CSV without target columns")
            points = load_points(file)
            if args.command == "predict":
                values = [predict_point(artifact, p) for p in points]
                write_csv(args.output, values)
                answer = {"saved": args.output, "total": len(values), "predicted": sum("power_w" in p for p in values)}
            else:
                costs = None
                if args.cost_column:
                    raw_candidates = read_rows(file)
                    if args.cost_column not in raw_candidates[0]:
                        raise ValueError("Requested relative-cost column is missing")
                    costs = {row["point_id"]: row[args.cost_column] for row in raw_candidates}
                answer = recommend(artifact, points, args.budget, args.strategy, costs)
                write_json(args.output, answer)
        elif args.command == "report":
            run = Path(args.run)
            output = args.output or str(run / "report.html")
            value = json.loads((run / "result.json").read_text(encoding="utf-8"))
            if value.get("kind") == "development_study":
                from .development import development_report

                development_report(value, value["oof_predictions"], output)
            elif value.get("kind") in {"learning_curve", "adaptive_replay"}:
                from .reporting import curve_report

                curve_report(value, output)
            else:
                report(value, output)
            answer = {"saved": output}
        else:
            from .demo import run_demo

            answer = run_demo(args.output)
        print(json.dumps(answer, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"octwin: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
