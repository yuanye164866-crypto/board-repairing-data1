from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run_evaluation import DEFAULT_RESULTS_FOLDER, run_evaluation
from evaluation.summarize_reports import format_latex, format_markdown, summarize_reports


CONFIGURATIONS: list[dict[str, Any]] = [
    {
        "label": "baseline",
        "description": "Sequential component grouping with the default contact graph, OCR, and groundplane handling.",
        "component_solver_mode": "sequential",
        "contact_solver_mode": "off",
        "ocr_enabled": True,
        "groundplane_enabled": True,
    },
    {
        "label": "global_component_solver",
        "description": "Global component candidate solver with the default contact graph.",
        "component_solver_mode": "global",
        "contact_solver_mode": "off",
        "ocr_enabled": True,
        "groundplane_enabled": True,
    },
    {
        "label": "contact_risk_averse",
        "description": "Risk-averse contact solver that deactivates high-risk contact edges.",
        "component_solver_mode": "sequential",
        "contact_solver_mode": "risk_averse",
        "ocr_enabled": True,
        "groundplane_enabled": True,
    },
    {
        "label": "contact_missing_edge",
        "description": "Risk-averse contact solver with positive missing-edge hypotheses.",
        "component_solver_mode": "sequential",
        "contact_solver_mode": "missing_edge",
        "ocr_enabled": True,
        "groundplane_enabled": True,
    },
    {
        "label": "no_ocr",
        "description": "OCR is disabled before the pipeline reaches Tesseract.",
        "component_solver_mode": "sequential",
        "contact_solver_mode": "off",
        "ocr_enabled": False,
        "groundplane_enabled": True,
    },
    {
        "label": "no_groundplane",
        "description": "Copper-plane and groundplane detection is disabled.",
        "component_solver_mode": "sequential",
        "contact_solver_mode": "off",
        "ocr_enabled": True,
        "groundplane_enabled": False,
    },
]

UNSUPPORTED_CONFIGURATIONS = [
    {
        "label": "no_via_pairing",
        "reason": "Via and pad pairing is part of the geometry stage and has no public switch.",
    },
    {
        "label": "hough_only_pads",
        "reason": "Pad fallback behavior is embedded in detection and is not parameterized.",
    },
    {
        "label": "no_relaxed_contact",
        "reason": "Relaxed trace-contact heuristics are internal thresholds, not a runtime mode.",
    },
]


def run_ablations(
    configurations: list[dict[str, Any]],
    output_folder: Path,
    samples: list[str] | None = None,
) -> dict[str, Any]:
    """Run selected ablation configurations and save comparison artifacts."""

    output_folder.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = []
    runs: list[dict[str, Any]] = []

    for index, configuration in enumerate(configurations, start=1):
        label = str(configuration["label"])
        print(f"[{index}/{len(configurations)}] Ablation: {label}")
        report, run_folder = run_evaluation(
            samples=samples,
            output_folder=output_folder / label,
            component_solver_mode=str(configuration["component_solver_mode"]),
            contact_solver_mode=str(configuration["contact_solver_mode"]),
            ocr_enabled=bool(configuration["ocr_enabled"]),
            groundplane_enabled=bool(configuration["groundplane_enabled"]),
            configuration_label=label,
        )
        report["ablation"] = {
            "label": label,
            "description": configuration.get("description", ""),
            "requested_configuration": configuration,
        }
        report_path = run_folder / "evaluation_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_paths.append(report_path)
        runs.append(
            {
                "label": label,
                "report": str(report_path),
                "result_folder": str(run_folder),
                "quality_status": (report.get("summary", {}) or {}).get("quality_status", ""),
            }
        )

    summary = summarize_reports(report_paths)
    manifest = {
        "schema": "prism.ablation_run",
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_folder": str(output_folder),
        "runs": runs,
        "unsupported_configurations": UNSUPPORTED_CONFIGURATIONS,
        "summary": summary,
    }
    (output_folder / "ablation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_folder / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_folder / "ablation_summary.md").write_text(
        format_markdown(summary) + unsupported_markdown(),
        encoding="utf-8",
    )
    (output_folder / "ablation_summary_table.tex").write_text(
        format_latex(summary),
        encoding="utf-8",
    )
    ablation_results = DEFAULT_RESULTS_FOLDER / "ablations"
    ablation_results.mkdir(parents=True, exist_ok=True)
    (ablation_results / "latest.txt").write_text(str(output_folder), encoding="utf-8")
    return manifest


def select_configurations(names: list[str]) -> list[dict[str, Any]]:
    """Resolve requested configuration labels."""

    if not names or "all" in names:
        return CONFIGURATIONS
    by_name = {str(configuration["label"]): configuration for configuration in CONFIGURATIONS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise SystemExit(
            "Unknown configuration(s): "
            + ", ".join(unknown)
            + ". Available configurations: "
            + ", ".join(sorted(by_name))
        )
    return [by_name[name] for name in names]


def unsupported_markdown() -> str:
    """Document intentionally unavailable ablations in the Markdown summary."""

    lines = ["", "## Configurations not run", ""]
    for item in UNSUPPORTED_CONFIGURATIONS:
        lines.append(f"- `{item['label']}`: {item['reason']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the ablation command-line workflow."""

    parser = argparse.ArgumentParser(
        description="Run supported PRISM synthetic ablations and generate JSON, Markdown, and LaTeX summaries."
    )
    parser.add_argument(
        "--configuration",
        action="append",
        default=[],
        help="Configuration label to run. Repeat for multiple configurations. Default: all.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Dataset sample to run. Repeat for multiple samples. Default: all.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Destination for all ablation runs and summaries.",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Return exit code zero even if one or more configurations fail quality thresholds.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = args.output_folder or (
        DEFAULT_RESULTS_FOLDER / "ablations" / timestamp
    )
    manifest = run_ablations(
        select_configurations(args.configuration),
        output_folder,
        samples=args.sample,
    )
    print(f"Manifest: {output_folder / 'ablation_manifest.json'}")
    print(f"Markdown summary: {output_folder / 'ablation_summary.md'}")
    statuses = [
        str(run.get("quality_status", ""))
        for run in manifest.get("runs", [])
    ]
    if any(status != "pass" for status in statuses) and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

