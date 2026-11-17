from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics import (
    assess_quality,
    match_electrical_objects,
    thresholds_for_sample,
)
from evaluation.summarize_reports import (
    format_latex,
    format_markdown,
    summarize_reports,
)


ROOT = Path(__file__).resolve().parents[1]


class EvaluationReferenceTests(unittest.TestCase):
    def test_all_dataset_samples_have_complete_reference_data(self) -> None:
        dataset_samples = {
            path.name.lower()
            for path in (ROOT / "Dataset").iterdir()
            if path.is_dir() and (path / "TOP.png").exists() and (path / "BOTTOM.png").exists()
        }
        reference_samples = {
            path.name
            for path in (ROOT / "evaluation" / "reference").iterdir()
            if path.is_dir()
        }
        self.assertEqual(dataset_samples, reference_samples)

        expected_files = {
            "expected_components.json",
            "expected_geometry.json",
            "expected_netlist.json",
            "expected_schematic_components.json",
            "expected_stats.json",
        }
        for sample in sorted(reference_samples):
            folder = ROOT / "evaluation" / "reference" / sample
            self.assertEqual(expected_files, {path.name for path in folder.glob("*.json")})
            for path in folder.glob("*.json"):
                json.loads(path.read_text(encoding="utf-8"))

    def test_reference_schematic_paths_are_repository_relative(self) -> None:
        for path in (ROOT / "evaluation" / "reference").rglob(
            "expected_schematic_components.json"
        ):
            data = json.loads(path.read_text(encoding="utf-8"))
            source = str(data["source_schematic"])
            self.assertTrue(source.startswith("Dataset/"), source)
            self.assertNotIn("\\", source)


class EvaluationMetricTests(unittest.TestCase):
    def test_geometry_matching_accepts_scaled_and_mirrored_points(self) -> None:
        expected = {
            "pads": [
                {"x": 10.0, "y": 10.0, "side": "both"},
                {"x": 90.0, "y": 10.0, "side": "both"},
                {"x": 50.0, "y": 80.0, "side": "both"},
            ],
            "vias": [{"at": [20.0, 50.0]}],
            "segments": [],
            "zones": [],
            "outline": {"bbox": [0.0, 100.0, 0.0, 100.0]},
        }
        actual = [
            {"x": 900.0, "y": 100.0},
            {"x": 100.0, "y": 100.0},
            {"x": 500.0, "y": 800.0},
            {"x": 800.0, "y": 500.0},
        ]
        reconstruction = {
            "parameters": {"top_rectification": {"size": [1000, 1000]}}
        }
        result = match_electrical_objects(expected, actual, reconstruction)
        self.assertEqual(3, result["matched_pads"])
        self.assertEqual(1, result["matched_vias"])

    def test_sample_thresholds_override_defaults(self) -> None:
        thresholds = thresholds_for_sample("Simple")
        self.assertEqual(1.0, thresholds["min_pad_recall"])
        self.assertEqual(0, thresholds["max_abs_net_count_delta"])

    def test_quality_failure_names_the_metric_and_threshold(self) -> None:
        result = {
            "status": "ok",
            "metrics": {
                "pad_recall": 0.5,
                "pad_precision": 1.0,
                "pair_recall": 1.0,
                "pair_precision": 1.0,
                "net_size_signature_recall": 1.0,
                "net_size_signature_precision": 1.0,
                "extra_physical_pad_candidates": 0,
                "net_count_delta": 0,
                "open_problem_count": 0,
            },
            "layout": {"symbol_collisions": 0},
        }
        quality = assess_quality("Simple", result)
        self.assertEqual("fail", quality["status"])
        self.assertTrue(any("pad_recall" in failure for failure in quality["failures"]))


class EvaluationSummaryTests(unittest.TestCase):
    def test_summary_generates_json_markdown_and_latex_views(self) -> None:
        report = {
            "configuration": {
                "label": "baseline",
                "component_solver_mode": "sequential",
                "contact_solver_mode": "off",
                "ocr_enabled": True,
                "groundplane_enabled": True,
            },
            "samples": [
                {
                    "sample": "simple",
                    "status": "ok",
                    "quality": {"status": "pass"},
                    "metrics": {
                        "pad_true_positives": 2,
                        "pad_false_positives": 0,
                        "pad_false_negatives": 0,
                        "matched_pairs": 2,
                        "expected_pairs": 2,
                        "actual_pairs": 2,
                        "pin_connection_true_positives": 2,
                        "pin_connection_false_positives": 0,
                        "pin_connection_false_negatives": 0,
                        "netlist_edit_distance": 0,
                        "expected_pin_connection_count": 2,
                        "matched_net_size_groups": 2,
                        "expected_net_size_groups": 2,
                        "actual_net_size_groups": 2,
                        "matched_component_references": 1,
                        "expected_component_references": ["R1"],
                        "actual_component_references": ["R1"],
                        "open_problem_count": 0,
                    },
                    "layout": {"symbol_collisions": 0},
                    "schematic": {"roundtrip_status": "pass"},
                    "ocr": {"text_count": 1},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "evaluation_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            summary = summarize_reports([report_path])

        row = summary["reports"][0]
        self.assertEqual(1.0, row["derived"]["pad_f1"])
        self.assertEqual(1.0, row["derived"]["pin_f1"])
        self.assertIn("baseline", format_markdown(summary))
        self.assertIn("\\begin{tabular}", format_latex(summary))


if __name__ == "__main__":
    unittest.main()

