from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "min_pad_recall": 0.75,
    "min_pad_precision": 0.70,
    "min_pair_recall": 0.75,
    "min_pair_precision": 0.75,
    "min_net_size_signature_recall": 0.50,
    "min_net_size_signature_precision": 0.50,
    "max_extra_physical_pad_candidates": 3,
    "max_abs_net_count_delta": 2,
    "max_open_problem_count": 1,
    "max_symbol_collisions": 0,
}

SAMPLE_THRESHOLDS: dict[str, dict[str, float | int]] = {
    "connector": {
        "min_pad_recall": 0.75,
        "min_pair_recall": 0.75,
        "min_net_size_signature_recall": 0.25,
        "min_net_size_signature_precision": 0.16,
        "max_abs_net_count_delta": 2,
        "max_open_problem_count": 0,
    },
    "crossover": {
        "min_pad_recall": 1.0,
        "min_pad_precision": 1.0,
        "min_pair_recall": 1.0,
        "min_pair_precision": 1.0,
        "min_net_size_signature_recall": 0.75,
        "min_net_size_signature_precision": 0.60,
        "max_extra_physical_pad_candidates": 0,
        "max_abs_net_count_delta": 1,
        "max_open_problem_count": 0,
    },
    "elements": {
        "min_pad_recall": 0.85,
        "min_pad_precision": 1.0,
        "min_pair_recall": 0.825,
        "min_pair_precision": 1.0,
        "min_net_size_signature_recall": 0.45,
        "min_net_size_signature_precision": 0.33,
        "max_extra_physical_pad_candidates": 0,
        "max_abs_net_count_delta": 4,
        "max_open_problem_count": 2,
    },
    "floating": {
        "min_pad_recall": 1.0,
        "min_pad_precision": 0.88,
        "min_pair_recall": 1.0,
        "min_pair_precision": 1.0,
        "min_net_size_signature_recall": 0.50,
        "min_net_size_signature_precision": 0.40,
        "max_extra_physical_pad_candidates": 1,
        "max_abs_net_count_delta": 1,
        "max_open_problem_count": 1,
    },
    "groundplane": {
        "min_pad_recall": 0.75,
        "min_pad_precision": 1.0,
        "min_pair_recall": 0.75,
        "min_pair_precision": 1.0,
        "max_extra_physical_pad_candidates": 0,
        "max_abs_net_count_delta": 1,
        "max_open_problem_count": 0,
    },
    "simple": {
        "min_pad_recall": 1.0,
        "min_pad_precision": 1.0,
        "min_pair_recall": 1.0,
        "min_pair_precision": 1.0,
        "min_net_size_signature_recall": 1.0,
        "min_net_size_signature_precision": 1.0,
        "max_extra_physical_pad_candidates": 0,
        "max_abs_net_count_delta": 0,
        "max_open_problem_count": 0,
    },
}


def evaluate_sample(reference_folder: Path, output_folder: Path, logs: list[str]) -> dict[str, Any]:
    """Compare one PRISM run with the versioned reference annotations."""

    work_folder = output_folder / "work"
    expected_stats = load_json(reference_folder / "expected_stats.json")
    expected_geometry = load_json(reference_folder / "expected_geometry.json")
    expected_components = load_json(reference_folder / "expected_components.json")
    expected_schematic_components = load_json(reference_folder / "expected_schematic_components.json")
    expected_netlist = load_json(reference_folder / "expected_netlist.json")

    actual_summary = load_json(work_folder / "summary_corrected.json")
    reconstruction = load_json(work_folder / "reconstruction.json")
    problems = load_json(work_folder / "problems.json", default=[])
    graph = load_json(work_folder / "connectivity_graph.json")
    evidence_dataset = load_json(work_folder / "evidence_dataset.json")
    contact_solver = load_json(work_folder / "contact_solver.json")
    layout = load_json(output_folder / "schematic.layout.json")
    roundtrip = load_json(output_folder / "schematic.roundtrip.json")

    metrics = calculate_metrics(
        expected_geometry=expected_geometry,
        expected_components=expected_components,
        expected_schematic_components=expected_schematic_components,
        expected_netlist=expected_netlist,
        actual_summary=actual_summary,
        reconstruction=reconstruction,
        problems=problems if isinstance(problems, list) else [],
    )
    return {
        "sample": reference_folder.name,
        "status": "ok",
        "output_folder": str(output_folder),
        "expected": expected_stats,
        "actual": actual_summary,
        "metrics": metrics,
        "problem_types": dict(
            sorted(
                Counter(
                    str(problem.get("type", ""))
                    for problem in problems
                    if isinstance(problem, dict) and problem.get("status") == "open"
                ).items()
            )
        ),
        "graph_summary": graph.get("summary", {}) if isinstance(graph, dict) else {},
        "evidence_dataset": summarize_optional_artifact(evidence_dataset),
        "contact_solver": summarize_optional_artifact(contact_solver),
        "layout": {
            "symbol_count": layout.get("symbol_count"),
            "symbol_collisions": int(layout.get("symbol_collisions", 0) or 0),
            "routing_conflict_count": int(layout.get("routing_conflict_count", 0) or 0),
            "wire_preflight_status": (layout.get("wire_preflight", {}) or {}).get("status", ""),
            "wire_preflight_conflict_count": int(
                (layout.get("wire_preflight", {}) or {}).get("wire_conflict_count", 0) or 0
            ),
            "label_nets": layout.get("label_nets", []),
            "wired_nets": layout.get("wired_nets", []),
        },
        "schematic": {
            "roundtrip_status": roundtrip.get("status", ""),
            "matched_group_count": int(roundtrip.get("matched_group_count", 0) or 0),
            "expected_group_count": int(roundtrip.get("expected_group_count", 0) or 0),
            "actual_group_count": int(roundtrip.get("actual_group_count", 0) or 0),
            "missing_group_count": len(roundtrip.get("missing_groups", []) or []),
            "extra_group_count": len(roundtrip.get("extra_groups", []) or []),
            "erc_violation_count": int((roundtrip.get("erc", {}) or {}).get("violation_count", 0) or 0),
        },
        "ocr": summarize_ocr(reconstruction, logs),
        "logs_tail": logs[-80:],
    }


def calculate_metrics(
    *,
    expected_geometry: dict[str, Any],
    expected_components: dict[str, Any],
    expected_schematic_components: dict[str, Any],
    expected_netlist: dict[str, Any],
    actual_summary: dict[str, Any],
    reconstruction: dict[str, Any],
    problems: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate geometry, connectivity, and component metrics for one sample."""

    expected_pads = list(expected_geometry.get("pads", []))
    expected_both_sides = [pad for pad in expected_pads if pad.get("side") == "both"]
    actual_pads = physical_actual_pads(reconstruction)
    matching = match_electrical_objects(expected_geometry, actual_pads, reconstruction)

    matched_pads = int(matching["matched_pads"])
    matched_vias = int(matching["matched_vias"])
    expected_pad_count = len(expected_pads)
    actual_pad_count = max(0, len(actual_pads) - matched_vias)
    pad_false_positives = max(0, actual_pad_count - matched_pads)
    pad_false_negatives = max(0, expected_pad_count - matched_pads)

    actual_pair_count = max(0, len(reconstruction.get("hole_pairs", [])) - matched_vias)
    expected_pair_count = len(expected_both_sides)
    matched_pairs = min(expected_pair_count, actual_pair_count)

    expected_net_sizes = sorted(
        int(net.get("pin_count", len(net.get("pins", []))) or 0)
        for net in expected_netlist.get("nets", [])
        if int(net.get("pin_count", len(net.get("pins", []))) or 0) > 0
    )
    actual_net_sizes = actual_physical_net_sizes(reconstruction)
    net_size_matched = multiset_overlap(expected_net_sizes, actual_net_sizes)

    expected_refs = sorted(
        str(component.get("ref", ""))
        for component in expected_schematic_components.get("components", [])
        if component.get("ref")
    )
    if not expected_refs:
        expected_refs = sorted(
            str(component.get("ref", ""))
            for component in expected_components.get("components", [])
            if component.get("ref")
        )
    actual_refs = sorted(
        str(element.get("ref", ""))
        for element in reconstruction.get("elements", [])
        if element.get("ref")
    )
    matched_refs = sorted(set(expected_refs) & set(actual_refs))

    expected_pin_connections = {
        str(pin)
        for net in expected_netlist.get("nets", [])
        for pin in net.get("pins", [])
        if pin
    }
    actual_pin_connections = {
        f"{element.get('ref')}.{pin}"
        for element in reconstruction.get("elements", [])
        for pin, net in (element.get("pins", {}) or {}).items()
        if element.get("ref") and net and net != "NET?"
    }
    pin_true_positives = len(expected_pin_connections & actual_pin_connections)
    pin_false_positives = len(actual_pin_connections - expected_pin_connections)
    pin_false_negatives = len(expected_pin_connections - actual_pin_connections)

    open_problems = [
        problem for problem in problems
        if isinstance(problem, dict) and problem.get("status") == "open"
    ]
    expected_net_count = len(expected_net_sizes)
    actual_net_count = len(actual_net_sizes)
    return {
        "expected_pads": expected_pad_count,
        "actual_physical_pad_candidates": actual_pad_count,
        "matched_pads": matched_pads,
        "matched_vias": matched_vias,
        "pad_true_positives": matched_pads,
        "pad_false_positives": pad_false_positives,
        "pad_false_negatives": pad_false_negatives,
        "pad_precision": ratio(matched_pads, matched_pads + pad_false_positives),
        "pad_recall": ratio(matched_pads, expected_pad_count),
        "extra_physical_pad_candidates": pad_false_positives,
        "expected_pairs": expected_pair_count,
        "actual_pairs": actual_pair_count,
        "matched_pairs": matched_pairs,
        "pair_precision": ratio(matched_pairs, actual_pair_count),
        "pair_recall": ratio(matched_pairs, expected_pair_count),
        "expected_net_size_groups": len(expected_net_sizes),
        "actual_net_size_groups": len(actual_net_sizes),
        "matched_net_size_groups": net_size_matched,
        "expected_net_size_signature": expected_net_sizes,
        "actual_net_size_signature": actual_net_sizes,
        "net_size_signature_precision": ratio(net_size_matched, len(actual_net_sizes)),
        "net_size_signature_recall": ratio(net_size_matched, len(expected_net_sizes)),
        "expected_net_count": expected_net_count,
        "actual_net_count": actual_net_count,
        "net_count_delta": actual_net_count - expected_net_count,
        "expected_component_references": expected_refs,
        "actual_component_references": actual_refs,
        "matched_component_references": len(matched_refs),
        "matched_component_reference_names": matched_refs,
        "component_reference_precision": ratio(len(matched_refs), len(actual_refs)),
        "component_reference_recall": ratio(len(matched_refs), len(expected_refs)),
        "expected_pin_connection_count": len(expected_pin_connections),
        "actual_pin_connection_count": len(actual_pin_connections),
        "pin_connection_true_positives": pin_true_positives,
        "pin_connection_false_positives": pin_false_positives,
        "pin_connection_false_negatives": pin_false_negatives,
        "pin_connection_precision": ratio(
            pin_true_positives,
            pin_true_positives + pin_false_positives,
        ),
        "pin_connection_recall": ratio(
            pin_true_positives,
            pin_true_positives + pin_false_negatives,
        ),
        "netlist_edit_distance": pin_false_positives + pin_false_negatives,
        "open_problem_count": len(open_problems),
        "open_problem_severity": dict(
            sorted(Counter(str(problem.get("severity", "unknown")) for problem in open_problems).items())
        ),
        "reported_element_count": int(actual_summary.get("element_count", 0) or 0),
    }


def assess_quality(sample: str, result: dict[str, Any]) -> dict[str, Any]:
    """Apply the versioned acceptance thresholds to one evaluation result."""

    thresholds = thresholds_for_sample(sample)
    failures: list[str] = []
    if result.get("status") != "ok":
        failures.append(f"pipeline status is {result.get('status', 'unknown')}")
        return {"status": "fail", "failures": failures, "thresholds": thresholds}

    metrics = result.get("metrics", {}) or {}
    layout = result.get("layout", {}) or {}
    check_minimum(metrics, "pad_recall", thresholds["min_pad_recall"], failures)
    check_minimum(metrics, "pad_precision", thresholds["min_pad_precision"], failures)
    check_minimum(metrics, "pair_recall", thresholds["min_pair_recall"], failures)
    check_minimum(metrics, "pair_precision", thresholds["min_pair_precision"], failures)
    check_minimum(
        metrics,
        "net_size_signature_recall",
        thresholds["min_net_size_signature_recall"],
        failures,
    )
    check_minimum(
        metrics,
        "net_size_signature_precision",
        thresholds["min_net_size_signature_precision"],
        failures,
    )
    check_maximum(
        metrics,
        "extra_physical_pad_candidates",
        thresholds["max_extra_physical_pad_candidates"],
        failures,
    )
    check_absolute_maximum(
        metrics,
        "net_count_delta",
        thresholds["max_abs_net_count_delta"],
        failures,
    )
    check_maximum(
        metrics,
        "open_problem_count",
        thresholds["max_open_problem_count"],
        failures,
    )
    check_maximum(
        layout,
        "symbol_collisions",
        thresholds["max_symbol_collisions"],
        failures,
    )
    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "thresholds": thresholds,
    }


def summarize_ocr(reconstruction: dict[str, Any], logs: list[str]) -> dict[str, Any]:
    """Summarize OCR activity without treating OCR as a graph ablation."""

    parameters = reconstruction.get("parameters", {}) or {}
    ocr = parameters.get("ocr", {}) or {}
    disabled = not bool(parameters.get("ocr_enabled", True))
    unavailable = any("Tesseract executable is missing" in line for line in logs)
    engine_errors = [line for line in logs if "OCR: engine error" in line]
    return {
        "enabled": not disabled,
        "disabled": disabled,
        "available": not disabled and not unavailable,
        "text_count": int(ocr.get("text_count", 0) or 0),
        "pass_count": int(ocr.get("pass_count", 0) or 0),
        "engine_error_count": len(engine_errors),
    }


def summarize_optional_artifact(data: Any) -> dict[str, Any]:
    """Return a compact description of an optional diagnostic JSON artifact."""

    if not isinstance(data, dict) or not data:
        return {"available": False}
    summary = {
        "available": True,
        "schema": data.get("schema", ""),
        "schema_version": data.get("schema_version", 0),
    }
    for key in ("algorithm", "candidate_count", "selected_count", "object_classes", "diagnostic_layers"):
        if key in data:
            summary[key] = data[key]
    return summary


def physical_actual_pads(reconstruction: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse paired TOP/BOTTOM detections into physical pad candidates."""

    top_pads = list(reconstruction.get("top_pads", []))
    bottom_pads = list(reconstruction.get("bottom_pads", []))
    paired_bottom = {
        str(pair.get("pad_bottom"))
        for pair in reconstruction.get("hole_pairs", [])
        if pair.get("pad_bottom")
    }
    return [
        *top_pads,
        *[
            pad for pad in bottom_pads
            if f"BOTTOM:{pad.get('identifier')}" not in paired_bottom
        ],
    ]


def actual_physical_net_sizes(reconstruction: dict[str, Any]) -> list[int]:
    """Return a sorted physical-pad count for every reconstructed net."""

    bottom_to_top = {
        str(pair.get("pad_bottom")): str(pair.get("pad_top"))
        for pair in reconstruction.get("hole_pairs", [])
        if pair.get("pad_bottom") and pair.get("pad_top")
    }
    sizes: list[int] = []
    for net in reconstruction.get("nets", []):
        physical_pads = {
            bottom_to_top.get(str(node), str(node))
            for node in net.get("pads", [])
            if node
        }
        if physical_pads:
            sizes.append(len(physical_pads))
    return sorted(sizes)


def match_electrical_objects(
    expected_geometry: dict[str, Any],
    actual_pads: list[dict[str, Any]],
    reconstruction: dict[str, Any],
) -> dict[str, int]:
    """Match expected pads and vias to detections after scale/orientation normalization."""

    expected = expected_electrical_points(expected_geometry)
    actual_points = [
        (float(pad.get("x", 0.0)), float(pad.get("y", 0.0)))
        for pad in actual_pads
    ]
    if not expected or not actual_points:
        return {"matched_total": 0, "matched_pads": 0, "matched_vias": 0}

    expected_points = [(float(item["x"]), float(item["y"])) for item in expected]
    expected_boxes = unique_boxes(
        [box_from_points(expected_points), geometry_box(expected_geometry)]
    )
    actual_boxes = [box_from_points(actual_points)]
    size = (
        reconstruction.get("parameters", {})
        .get("top_rectification", {})
        .get("size", [])
    )
    if isinstance(size, list) and len(size) >= 2:
        width, height = float(size[0]), float(size[1])
        if width > 0 and height > 0:
            actual_boxes.append((0.0, width, 0.0, height))
    actual_boxes = unique_boxes(actual_boxes)

    best = {"matched_total": 0, "matched_pads": 0, "matched_vias": 0}
    for expected_box in expected_boxes:
        normalized_expected = normalize_points_in_box(expected_points, expected_box)
        for actual_box in actual_boxes:
            normalized_actual = normalize_points_in_box(actual_points, actual_box)
            for variant in normalized_point_variants(normalized_actual):
                result = match_with_kind_priority(
                    expected,
                    normalized_expected,
                    variant,
                    tolerance=0.075,
                )
                if (
                    result["matched_pads"],
                    result["matched_total"],
                    result["matched_vias"],
                ) > (
                    best["matched_pads"],
                    best["matched_total"],
                    best["matched_vias"],
                ):
                    best = result
    return best


def expected_electrical_points(expected_geometry: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract expected pad and via centers from reference geometry."""

    points = [
        {"kind": "pad", "x": float(pad.get("x", 0.0)), "y": float(pad.get("y", 0.0))}
        for pad in expected_geometry.get("pads", [])
    ]
    for via in expected_geometry.get("vias", []):
        at = via.get("at", [])
        if isinstance(at, list) and len(at) >= 2:
            points.append({"kind": "via", "x": float(at[0]), "y": float(at[1])})
    return points


def geometry_box(expected_geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    """Build a reference bounding box from all available KiCad geometry."""

    points: list[tuple[float, float]] = []
    for pad in expected_geometry.get("pads", []):
        points.append((float(pad.get("x", 0.0)), float(pad.get("y", 0.0))))
    for via in expected_geometry.get("vias", []):
        at = via.get("at", [])
        if isinstance(at, list) and len(at) >= 2:
            points.append((float(at[0]), float(at[1])))
    for segment in expected_geometry.get("segments", []):
        for key in ("start", "end"):
            point = segment.get(key, [])
            if isinstance(point, list) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    for zone in expected_geometry.get("zones", []):
        for point in zone.get("polygon", []):
            if isinstance(point, list) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    outline = expected_geometry.get("outline", {})
    bbox = outline.get("bbox", []) if isinstance(outline, dict) else []
    if isinstance(bbox, list) and len(bbox) >= 4:
        min_x, max_x, min_y, max_y = map(float, bbox[:4])
        points.extend([(min_x, min_y), (max_x, max_y)])
    return box_from_points(points)


def box_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Return min/max bounds for a point collection."""

    if not points:
        return 0.0, 1.0, 0.0, 1.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def unique_boxes(
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Remove degenerate and duplicate normalization boxes."""

    result: list[tuple[float, float, float, float]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        min_x, max_x, min_y, max_y = box
        if max_x - min_x <= 1e-6 or max_y - min_y <= 1e-6:
            continue
        key = tuple(round(value * 1000) for value in box)
        if key in seen:
            continue
        seen.add(key)
        result.append(box)
    return result or [(0.0, 1.0, 0.0, 1.0)]


def normalize_points_in_box(
    points: list[tuple[float, float]],
    box: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Normalize points to a unit square using a selected bounding box."""

    min_x, max_x, min_y, max_y = box
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    return [((x - min_x) / width, (y - min_y) / height) for x, y in points]


def normalized_point_variants(
    points: list[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """Return mirrored and axis-swapped variants used for image-side matching."""

    transforms = [
        lambda x, y: (x, y),
        lambda x, y: (1.0 - x, y),
        lambda x, y: (x, 1.0 - y),
        lambda x, y: (1.0 - x, 1.0 - y),
        lambda x, y: (y, x),
        lambda x, y: (1.0 - y, x),
        lambda x, y: (y, 1.0 - x),
        lambda x, y: (1.0 - y, 1.0 - x),
    ]
    return [[transform(x, y) for x, y in points] for transform in transforms]


def match_with_kind_priority(
    expected: list[dict[str, Any]],
    expected_points: list[tuple[float, float]],
    actual_points: list[tuple[float, float]],
    tolerance: float,
) -> dict[str, int]:
    """Greedily match pads before vias so vias do not consume component pads."""

    pad_indices = [index for index, item in enumerate(expected) if item.get("kind") == "pad"]
    via_indices = [index for index, item in enumerate(expected) if item.get("kind") == "via"]
    available_actual = list(range(len(actual_points)))
    pad_matches = greedy_match_subset(
        expected_points,
        actual_points,
        tolerance,
        pad_indices,
        available_actual,
    )
    occupied_actual = {actual_index for _expected_index, actual_index in pad_matches}
    via_matches = greedy_match_subset(
        expected_points,
        actual_points,
        tolerance,
        via_indices,
        [index for index in available_actual if index not in occupied_actual],
    )
    return {
        "matched_total": len(pad_matches) + len(via_matches),
        "matched_pads": len(pad_matches),
        "matched_vias": len(via_matches),
    }


def greedy_match_subset(
    expected_points: list[tuple[float, float]],
    actual_points: list[tuple[float, float]],
    tolerance: float,
    expected_indices: list[int],
    actual_indices: list[int],
) -> list[tuple[int, int]]:
    """Return non-overlapping point matches ordered by increasing distance."""

    candidates: list[tuple[float, int, int]] = []
    for expected_index in expected_indices:
        expected = expected_points[expected_index]
        for actual_index in actual_indices:
            actual = actual_points[actual_index]
            distance = math.hypot(expected[0] - actual[0], expected[1] - actual[1])
            if distance <= tolerance:
                candidates.append((distance, expected_index, actual_index))
    candidates.sort()
    used_expected: set[int] = set()
    used_actual: set[int] = set()
    result: list[tuple[int, int]] = []
    for _distance, expected_index, actual_index in candidates:
        if expected_index in used_expected or actual_index in used_actual:
            continue
        used_expected.add(expected_index)
        used_actual.add(actual_index)
        result.append((expected_index, actual_index))
    return result


def thresholds_for_sample(sample: str) -> dict[str, float | int]:
    """Merge default and sample-specific acceptance thresholds."""

    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(SAMPLE_THRESHOLDS.get(sample.lower(), {}))
    return thresholds


def check_minimum(
    data: dict[str, Any],
    key: str,
    threshold: float | int,
    failures: list[str],
) -> None:
    value = float(data.get(key, 0.0) or 0.0)
    if value < float(threshold):
        failures.append(f"{key}={value:.4g} is below {float(threshold):.4g}")


def check_maximum(
    data: dict[str, Any],
    key: str,
    threshold: float | int,
    failures: list[str],
) -> None:
    value = float(data.get(key, 0.0) or 0.0)
    if value > float(threshold):
        failures.append(f"{key}={value:.4g} exceeds {float(threshold):.4g}")


def check_absolute_maximum(
    data: dict[str, Any],
    key: str,
    threshold: float | int,
    failures: list[str],
) -> None:
    value = float(data.get(key, 0.0) or 0.0)
    if abs(value) > float(threshold):
        failures.append(f"|{key}|={abs(value):.4g} exceeds {float(threshold):.4g}")


def multiset_overlap(expected: list[int], actual: list[int]) -> int:
    """Count the overlap between two integer multisets."""

    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    return sum(min(expected_counts[value], actual_counts[value]) for value in expected_counts)


def ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a stable ratio, using one for two empty sets."""

    if float(denominator) <= 0:
        return 1.0 if float(numerator) == 0 else 0.0
    return round(float(numerator) / float(denominator), 6)


def load_json(path: Path, default: Any | None = None) -> Any:
    """Read a UTF-8 JSON file or return a caller-provided default."""

    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))

