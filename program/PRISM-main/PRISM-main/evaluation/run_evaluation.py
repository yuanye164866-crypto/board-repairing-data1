from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import assess_quality, evaluate_sample
from schematic_generator.analysis import run_analysis
from schematic_generator.automatic import (
    AUTOMATIC_FILTER_COMBINATION_COUNT,
    AUTOMATIC_GROUNDPLANE,
    AUTOMATIC_THRESHOLD_ALIGNMENT,
    samples_colors_automatic,
)
from schematic_generator.platform_support import find_kicad_cli, find_tesseract


DATASET_FOLDER = ROOT / "Dataset"
REFERENCE_FOLDER = ROOT / "evaluation" / "reference"
DEFAULT_RESULTS_FOLDER = ROOT / "evaluation" / "results"


def run_evaluation(
    *,
    samples: list[str] | None = None,
    output_folder: Path | None = None,
    component_solver_mode: str = "sequential",
    contact_solver_mode: str = "off",
    ocr_enabled: bool = True,
    groundplane_enabled: bool = AUTOMATIC_GROUNDPLANE,
    tesseract_path: Path | None = None,
    filter_combination_count: int = AUTOMATIC_FILTER_COMBINATION_COUNT,
    alignment_threshold: float = AUTOMATIC_THRESHOLD_ALIGNMENT,
    configuration_label: str = "",
) -> tuple[dict[str, Any], Path]:
    """Run selected synthetic samples and save a machine-readable evaluation report."""

    selected = select_samples(samples)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = output_folder or (DEFAULT_RESULTS_FOLDER / timestamp)
    result_folder.mkdir(parents=True, exist_ok=True)
    runs_folder = result_folder / "runs"
    runs_folder.mkdir(parents=True, exist_ok=True)

    resolved_tesseract = tesseract_path if ocr_enabled else None
    if ocr_enabled and resolved_tesseract is None:
        resolved_tesseract = find_tesseract()

    report: dict[str, Any] = {
        "schema": "prism.synthetic_evaluation",
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configuration": {
            "label": configuration_label or build_configuration_label(
                component_solver_mode,
                contact_solver_mode,
                ocr_enabled,
                groundplane_enabled,
            ),
            "component_solver_mode": component_solver_mode,
            "contact_solver_mode": contact_solver_mode,
            "ocr_enabled": bool(ocr_enabled),
            "groundplane_enabled": bool(groundplane_enabled),
            "filter_combination_count": int(filter_combination_count),
            "alignment_threshold": float(alignment_threshold),
        },
        "environment": environment_manifest(resolved_tesseract),
        "dataset_folder": str(DATASET_FOLDER.relative_to(ROOT)),
        "reference_folder": str(REFERENCE_FOLDER.relative_to(ROOT)),
        "result_folder": str(result_folder),
        "samples": [],
    }

    for index, sample in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] Evaluating {sample.name}")
        logs: list[str] = []
        try:
            analysis_report = run_analysis(
                sample / "TOP.png",
                sample / "BOTTOM.png",
                runs_folder,
                filter_combination_count=filter_combination_count,
                alignment_threshold=alignment_threshold,
                schematic_name=sample.name.lower(),
                tesseract_path=resolved_tesseract,
                color_samples=samples_colors_automatic(
                    sample / "TOP.png",
                    sample / "BOTTOM.png",
                ),
                has_groundplane=groundplane_enabled,
                component_solver_mode=component_solver_mode,
                contact_solver_mode=contact_solver_mode,
                ocr_enabled=ocr_enabled,
                log=logs.append,
            )
            sample_result = evaluate_sample(
                REFERENCE_FOLDER / sample.name.lower(),
                Path(analysis_report.output_folder),
                logs,
            )
        except Exception as error:
            sample_result = {
                "sample": sample.name.lower(),
                "status": "error",
                "error": str(error),
                "logs_tail": logs[-80:],
            }
        sample_result["quality"] = assess_quality(sample.name, sample_result)
        report["samples"].append(sample_result)

    report["summary"] = summarize_run(report["samples"])
    write_report(result_folder, report)
    DEFAULT_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    (DEFAULT_RESULTS_FOLDER / "latest.txt").write_text(str(result_folder), encoding="utf-8")
    return report, result_folder


def select_samples(names: list[str] | None) -> list[Path]:
    """Resolve case-insensitive sample names against the versioned dataset."""

    available = {
        path.name.lower(): path
        for path in DATASET_FOLDER.iterdir()
        if path.is_dir() and (path / "TOP.png").exists() and (path / "BOTTOM.png").exists()
    }
    if not names or any(name.lower() == "all" for name in names):
        selected = [available[name] for name in sorted(available)]
    else:
        unknown = sorted({name for name in names if name.lower() not in available})
        if unknown:
            raise SystemExit(
                "Unknown sample(s): "
                + ", ".join(unknown)
                + ". Available samples: "
                + ", ".join(sorted(available))
            )
        selected = [available[name.lower()] for name in names]

    missing_reference = [
        path.name for path in selected
        if not (REFERENCE_FOLDER / path.name.lower() / "expected_stats.json").exists()
    ]
    if missing_reference:
        raise SystemExit(
            "Missing evaluation reference data for: " + ", ".join(missing_reference)
        )
    return selected


def summarize_run(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate top-level pass/fail counts for an evaluation run."""

    pipeline_errors = [
        str(sample.get("sample", ""))
        for sample in samples
        if sample.get("status") != "ok"
    ]
    quality_failures = [
        str(sample.get("sample", ""))
        for sample in samples
        if (sample.get("quality", {}) or {}).get("status") != "pass"
    ]
    return {
        "sample_count": len(samples),
        "pipeline_error_count": len(pipeline_errors),
        "pipeline_errors": pipeline_errors,
        "quality_pass_count": len(samples) - len(quality_failures),
        "quality_failure_count": len(quality_failures),
        "quality_failures": quality_failures,
        "quality_status": "fail" if pipeline_errors or quality_failures else "pass",
    }


def environment_manifest(tesseract_path: Path | None) -> dict[str, Any]:
    """Record software versions needed to interpret or reproduce a run."""

    kicad_cli = find_kicad_cli()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "prism": package_version("prism-pcb"),
        "numpy": package_version("numpy"),
        "opencv_python": package_version("opencv-python"),
        "pillow": package_version("Pillow"),
        "pytesseract": package_version("pytesseract"),
        "tesseract_path": str(tesseract_path) if tesseract_path else "",
        "tesseract_available": bool(tesseract_path),
        "kicad_cli_path": str(kicad_cli) if kicad_cli else "",
        "kicad_cli_available": bool(kicad_cli),
    }


def package_version(package: str) -> str:
    """Return an installed package version without making it a hard requirement."""

    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def build_configuration_label(
    component_solver_mode: str,
    contact_solver_mode: str,
    ocr_enabled: bool,
    groundplane_enabled: bool,
) -> str:
    """Build a stable label from the main evaluation switches."""

    return "+".join(
        [
            component_solver_mode,
            contact_solver_mode,
            "ocr_on" if ocr_enabled else "ocr_off",
            "groundplane_on" if groundplane_enabled else "groundplane_off",
        ]
    )


def write_report(folder: Path, report: dict[str, Any]) -> None:
    """Write JSON and Markdown views of one evaluation run."""

    (folder / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (folder / "evaluation_report.md").write_text(
        format_markdown_report(report),
        encoding="utf-8",
    )


def format_markdown_report(report: dict[str, Any]) -> str:
    """Render a concise human-readable evaluation report."""

    configuration = report.get("configuration", {}) or {}
    summary = report.get("summary", {}) or {}
    lines = [
        "# PRISM synthetic evaluation",
        "",
        f"- Configuration: `{configuration.get('label', '')}`",
        f"- Samples: {summary.get('sample_count', 0)}",
        f"- Quality status: **{summary.get('quality_status', 'unknown')}**",
        f"- Pipeline errors: {summary.get('pipeline_error_count', 0)}",
        f"- Quality passes: {summary.get('quality_pass_count', 0)}/{summary.get('sample_count', 0)}",
        "",
        "| Sample | Pipeline | Quality | Pad P/R | Pair P/R | Net-size P/R | Pin TP/FP/FN | Open problems |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for sample in report.get("samples", []):
        metrics = sample.get("metrics", {}) or {}
        quality = sample.get("quality", {}) or {}
        lines.append(
            "| {sample} | {pipeline} | {quality} | {pad_p}/{pad_r} | "
            "{pair_p}/{pair_r} | {net_p}/{net_r} | {tp}/{fp}/{fn} | {open_count} |".format(
                sample=sample.get("sample", ""),
                pipeline=sample.get("status", ""),
                quality=quality.get("status", ""),
                pad_p=percent(metrics.get("pad_precision")),
                pad_r=percent(metrics.get("pad_recall")),
                pair_p=percent(metrics.get("pair_precision")),
                pair_r=percent(metrics.get("pair_recall")),
                net_p=percent(metrics.get("net_size_signature_precision")),
                net_r=percent(metrics.get("net_size_signature_recall")),
                tp=metrics.get("pin_connection_true_positives", 0),
                fp=metrics.get("pin_connection_false_positives", 0),
                fn=metrics.get("pin_connection_false_negatives", 0),
                open_count=metrics.get("open_problem_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "A failed quality threshold is a regression signal for this controlled synthetic dataset.",
            "It is not a claim about performance on photographs of physical PCBs.",
            "",
        ]
    )
    return "\n".join(lines)


def percent(value: Any) -> str:
    """Format a ratio as a percentage for Markdown reports."""

    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for synthetic evaluation."""

    parser = argparse.ArgumentParser(
        description="Run PRISM on the versioned synthetic PCB dataset and compare the outputs with reference annotations."
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Sample name to run. Repeat the option for multiple samples. Default: all.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Destination for reports and per-sample pipeline outputs.",
    )
    parser.add_argument(
        "--component-solver",
        choices=("sequential", "global"),
        default="sequential",
        help="Component inference mode.",
    )
    parser.add_argument(
        "--contact-solver",
        choices=("off", "risk_averse", "missing_edge"),
        default="off",
        help="Contact inference mode.",
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Skip OCR and do not invoke Tesseract.",
    )
    parser.add_argument(
        "--no-groundplane",
        dest="groundplane_enabled",
        action="store_false",
        help="Disable copper-plane detection.",
    )
    parser.add_argument(
        "--tesseract-path",
        type=Path,
        help="Explicit Tesseract executable used when OCR is enabled.",
    )
    parser.add_argument(
        "--filter-combinations",
        type=int,
        default=AUTOMATIC_FILTER_COMBINATION_COUNT,
        help=f"Number of trace-mask filter combinations (default: {AUTOMATIC_FILTER_COMBINATION_COUNT}).",
    )
    parser.add_argument(
        "--alignment-threshold",
        type=float,
        default=AUTOMATIC_THRESHOLD_ALIGNMENT,
        help=f"Minimum alignment score (default: {AUTOMATIC_THRESHOLD_ALIGNMENT}).",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Write reports but return exit code zero when quality thresholds fail.",
    )
    parser.set_defaults(groundplane_enabled=AUTOMATIC_GROUNDPLANE)
    return parser


def main() -> int:
    """Run the command-line evaluation workflow."""

    args = build_parser().parse_args()
    report, result_folder = run_evaluation(
        samples=args.sample,
        output_folder=args.output_folder,
        component_solver_mode=args.component_solver,
        contact_solver_mode=args.contact_solver,
        ocr_enabled=not args.disable_ocr,
        groundplane_enabled=args.groundplane_enabled,
        tesseract_path=args.tesseract_path,
        filter_combination_count=args.filter_combinations,
        alignment_threshold=args.alignment_threshold,
    )
    print(f"JSON report: {result_folder / 'evaluation_report.json'}")
    print(f"Markdown report: {result_folder / 'evaluation_report.md'}")
    print(f"Quality status: {report['summary']['quality_status']}")
    if report["summary"]["quality_status"] != "pass" and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

