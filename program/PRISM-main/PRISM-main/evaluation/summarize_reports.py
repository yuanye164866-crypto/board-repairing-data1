from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 evaluation report."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_report(
    path: str | Path,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate one evaluation report into a row suitable for comparison."""

    report_path = Path(path)
    data = report if report is not None else load_report(report_path)
    samples = data.get("samples", []) if isinstance(data.get("samples", []), list) else []
    totals = {
        "sample_count": len(samples),
        "pipeline_ok_count": 0,
        "quality_pass_count": 0,
        "roundtrip_pass_count": 0,
        "roundtrip_checked_count": 0,
        "open_problem_count": 0,
        "symbol_collision_count": 0,
        "ocr_text_count": 0,
        "pad_tp": 0,
        "pad_fp": 0,
        "pad_fn": 0,
        "pair_tp": 0,
        "pair_fp": 0,
        "pair_fn": 0,
        "pin_tp": 0,
        "pin_fp": 0,
        "pin_fn": 0,
        "netlist_edit_distance": 0,
        "expected_pin_connections": 0,
        "net_size_matched": 0,
        "net_size_expected": 0,
        "net_size_actual": 0,
        "matched_component_refs": 0,
        "expected_component_refs": 0,
        "actual_component_refs": 0,
    }

    sample_rows: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        metrics = sample.get("metrics", {}) or {}
        layout = sample.get("layout", {}) or {}
        schematic = sample.get("schematic", {}) or {}
        quality = sample.get("quality", {}) or {}

        if sample.get("status") == "ok":
            totals["pipeline_ok_count"] += 1
        if quality.get("status") == "pass":
            totals["quality_pass_count"] += 1

        roundtrip = str(schematic.get("roundtrip_status", "") or "")
        if roundtrip:
            totals["roundtrip_checked_count"] += 1
        if roundtrip == "pass":
            totals["roundtrip_pass_count"] += 1

        totals["symbol_collision_count"] += int(layout.get("symbol_collisions", 0) or 0)
        totals["ocr_text_count"] += int((sample.get("ocr", {}) or {}).get("text_count", 0) or 0)
        totals["open_problem_count"] += int(metrics.get("open_problem_count", 0) or 0)

        add_metric(totals, "pad_tp", metrics, "pad_true_positives")
        add_metric(totals, "pad_fp", metrics, "pad_false_positives")
        add_metric(totals, "pad_fn", metrics, "pad_false_negatives")
        pair_tp = int(metrics.get("matched_pairs", 0) or 0)
        expected_pairs = int(metrics.get("expected_pairs", 0) or 0)
        actual_pairs = int(metrics.get("actual_pairs", 0) or 0)
        totals["pair_tp"] += pair_tp
        totals["pair_fp"] += max(0, actual_pairs - pair_tp)
        totals["pair_fn"] += max(0, expected_pairs - pair_tp)
        add_metric(totals, "pin_tp", metrics, "pin_connection_true_positives")
        add_metric(totals, "pin_fp", metrics, "pin_connection_false_positives")
        add_metric(totals, "pin_fn", metrics, "pin_connection_false_negatives")
        add_metric(totals, "netlist_edit_distance", metrics, "netlist_edit_distance")
        add_metric(totals, "expected_pin_connections", metrics, "expected_pin_connection_count")
        add_metric(totals, "net_size_matched", metrics, "matched_net_size_groups")
        add_metric(totals, "net_size_expected", metrics, "expected_net_size_groups")
        add_metric(totals, "net_size_actual", metrics, "actual_net_size_groups")
        add_metric(totals, "matched_component_refs", metrics, "matched_component_references")
        expected_refs = metrics.get("expected_component_references", []) or []
        actual_refs = metrics.get("actual_component_references", []) or []
        totals["expected_component_refs"] += len(expected_refs) if isinstance(expected_refs, list) else 0
        totals["actual_component_refs"] += len(actual_refs) if isinstance(actual_refs, list) else 0

        sample_rows.append(
            {
                "sample": sample.get("sample", ""),
                "pipeline": sample.get("status", ""),
                "quality": quality.get("status", ""),
                "roundtrip": roundtrip,
                "pin_tp": int(metrics.get("pin_connection_true_positives", 0) or 0),
                "pin_fp": int(metrics.get("pin_connection_false_positives", 0) or 0),
                "pin_fn": int(metrics.get("pin_connection_false_negatives", 0) or 0),
                "netlist_edit_distance": int(metrics.get("netlist_edit_distance", 0) or 0),
                "open_problem_count": int(metrics.get("open_problem_count", 0) or 0),
            }
        )

    derived = {
        "pad_precision": ratio(totals["pad_tp"], totals["pad_tp"] + totals["pad_fp"]),
        "pad_recall": ratio(totals["pad_tp"], totals["pad_tp"] + totals["pad_fn"]),
        "pad_f1": f1(totals["pad_tp"], totals["pad_fp"], totals["pad_fn"]),
        "pair_precision": ratio(totals["pair_tp"], totals["pair_tp"] + totals["pair_fp"]),
        "pair_recall": ratio(totals["pair_tp"], totals["pair_tp"] + totals["pair_fn"]),
        "pair_f1": f1(totals["pair_tp"], totals["pair_fp"], totals["pair_fn"]),
        "pin_precision": ratio(totals["pin_tp"], totals["pin_tp"] + totals["pin_fp"]),
        "pin_recall": ratio(totals["pin_tp"], totals["pin_tp"] + totals["pin_fn"]),
        "pin_f1": f1(totals["pin_tp"], totals["pin_fp"], totals["pin_fn"]),
        "normalized_netlist_edit_distance": ratio(
            totals["netlist_edit_distance"],
            totals["expected_pin_connections"],
        ),
        "net_size_precision": ratio(totals["net_size_matched"], totals["net_size_actual"]),
        "net_size_recall": ratio(totals["net_size_matched"], totals["net_size_expected"]),
        "net_size_f1": f1_from_precision_recall(
            ratio(totals["net_size_matched"], totals["net_size_actual"]),
            ratio(totals["net_size_matched"], totals["net_size_expected"]),
        ),
        "component_reference_precision": ratio(
            totals["matched_component_refs"],
            totals["actual_component_refs"],
        ),
        "component_reference_recall": ratio(
            totals["matched_component_refs"],
            totals["expected_component_refs"],
        ),
    }
    configuration = data.get("configuration", {}) or {}
    return {
        "label": configuration.get("label", report_path.parent.name),
        "source_report": str(report_path),
        "configuration": configuration,
        "totals": totals,
        "derived": derived,
        "samples": sample_rows,
    }


def summarize_reports(paths: list[str | Path]) -> dict[str, Any]:
    """Summarize multiple evaluation reports into one versioned document."""

    return {
        "schema": "prism.evaluation_summary",
        "schema_version": 1,
        "reports": [summarize_report(path) for path in paths],
    }


def format_markdown(summary: dict[str, Any]) -> str:
    """Render a compact Markdown comparison table."""

    lines = [
        "# PRISM evaluation summary",
        "",
        "| Configuration | Component solver | Contact solver | OCR | Groundplane | Samples | Quality | Round-trip | Pad P/R | Pair P/R | Pin TP/FP/FN | Pin F1 | NED | Net-size F1 | Open |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("reports", []):
        configuration = row.get("configuration", {}) or {}
        totals = row.get("totals", {}) or {}
        derived = row.get("derived", {}) or {}
        lines.append(
            "| {label} | {component} | {contact} | {ocr} | {plane} | {samples} | "
            "{quality}/{samples} | {roundtrip}/{checked} | {pad_p}/{pad_r} | "
            "{pair_p}/{pair_r} | {tp}/{fp}/{fn} | {pin_f1} | {ned} | {net_f1} | {open_count} |".format(
                label=row.get("label", ""),
                component=configuration.get("component_solver_mode", ""),
                contact=configuration.get("contact_solver_mode", ""),
                ocr="on" if configuration.get("ocr_enabled", True) else "off",
                plane="on" if configuration.get("groundplane_enabled", True) else "off",
                samples=totals.get("sample_count", 0),
                quality=totals.get("quality_pass_count", 0),
                roundtrip=totals.get("roundtrip_pass_count", 0),
                checked=totals.get("roundtrip_checked_count", 0),
                pad_p=percent(derived.get("pad_precision")),
                pad_r=percent(derived.get("pad_recall")),
                pair_p=percent(derived.get("pair_precision")),
                pair_r=percent(derived.get("pair_recall")),
                tp=totals.get("pin_tp", 0),
                fp=totals.get("pin_fp", 0),
                fn=totals.get("pin_fn", 0),
                pin_f1=percent(derived.get("pin_f1")),
                ned=totals.get("netlist_edit_distance", 0),
                net_f1=percent(derived.get("net_size_f1")),
                open_count=totals.get("open_problem_count", 0),
            )
        )
    return "\n".join(lines) + "\n"


def format_latex(summary: dict[str, Any]) -> str:
    """Render publication-ready LaTeX rows from an evaluation summary."""

    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Configuration & RT pass & Pad F1 [\\%] & Pair F1 [\\%] & Pin F1 [\\%] & NED & Open \\\\",
        "\\midrule",
    ]
    for row in summary.get("reports", []):
        totals = row.get("totals", {}) or {}
        derived = row.get("derived", {}) or {}
        lines.append(
            "{} & {}/{} & {} & {} & {} & {} & {} \\\\".format(
                latex_escape(str(row.get("label", ""))),
                totals.get("roundtrip_pass_count", 0),
                totals.get("roundtrip_checked_count", 0),
                percent(derived.get("pad_f1")),
                percent(derived.get("pair_f1")),
                percent(derived.get("pin_f1")),
                totals.get("netlist_edit_distance", 0),
                totals.get("open_problem_count", 0),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def add_metric(
    totals: dict[str, int],
    target_key: str,
    data: dict[str, Any],
    source_key: str,
) -> None:
    """Add one integer metric to an aggregate total."""

    totals[target_key] += int(data.get(source_key, 0) or 0)


def ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a stable ratio for aggregate metrics."""

    return round(float(numerator) / float(denominator), 6) if float(denominator) else 0.0


def f1(true_positives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate an F1 score from confusion counts."""

    return f1_from_precision_recall(
        ratio(true_positives, true_positives + false_positives),
        ratio(true_positives, true_positives + false_negatives),
    )


def f1_from_precision_recall(precision: float, recall: float) -> float:
    """Calculate an F1 score from precision and recall."""

    if precision + recall <= 0:
        return 0.0
    return round(2.0 * precision * recall / (precision + recall), 6)


def percent(value: Any) -> str:
    """Format a ratio as a percentage without a percent sign."""

    try:
        return f"{float(value) * 100.0:.1f}"
    except (TypeError, ValueError):
        return "0.0"


def latex_escape(value: str) -> str:
    """Escape common LaTeX-special characters in configuration labels."""

    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def main() -> int:
    """Summarize report paths supplied on the command line."""

    parser = argparse.ArgumentParser(
        description="Summarize PRISM synthetic evaluation reports."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Paths to evaluation_report.json files.",
    )
    parser.add_argument("--output", type=Path, help="JSON summary output path.")
    parser.add_argument("--markdown", type=Path, help="Markdown summary output path.")
    parser.add_argument("--latex", type=Path, help="LaTeX table output path.")
    args = parser.parse_args()

    summary = summarize_reports(args.reports)
    markdown = format_markdown(summary)
    print(markdown)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(format_latex(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

