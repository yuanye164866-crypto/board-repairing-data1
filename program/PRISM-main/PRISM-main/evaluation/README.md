# PRISM Synthetic Evaluation

This directory contains the reproducible evaluation workflow used to exercise
PRISM on the controlled synthetic PCB samples in [`Dataset/`](../Dataset/).
It is intended for regression testing and ablation studies of the current
proof of concept. It does not measure performance on photographs of physical
PCBs and must not be interpreted as validation on real boards.

## Contents

```text
evaluation/
|-- reference/                 Versioned reference annotations for 12 samples
|-- metrics.py                 Matching, metrics, and quality thresholds
|-- run_evaluation.py          Single-configuration evaluation runner
|-- run_ablations.py           Six-configuration ablation runner
|-- summarize_reports.py       JSON, Markdown, and LaTeX report summarizer
`-- README.md                  This guide
```

Generated data is written to `evaluation/results/`, which is excluded from
version control.

## Requirements

Create the environment described in the main [`README.md`](../README.md), then
run all commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Tesseract is required when OCR is enabled. KiCad with `kicad-cli` is optional,
but without it the schematic round-trip and ERC fields may be reported as
unavailable. Both executable paths and the installed Python package versions
are recorded in every JSON report.

## Quick Check

Run one sample without OCR:

```bash
python evaluation/run_evaluation.py --sample Simple --disable-ocr --allow-fail
```

`--allow-fail` is useful for exploratory work: reports are always written, but
a failed quality threshold does not produce a non-zero process exit code.
Without this option, a quality regression returns exit code 1 and can be used
in automation.

## Baseline Evaluation

Run the default baseline on all 12 samples:

```bash
python evaluation/run_evaluation.py
```

Run selected samples:

```bash
python evaluation/run_evaluation.py --sample Simple --sample Crossover
```

Use an explicit Tesseract executable:

```powershell
python evaluation/run_evaluation.py `
  --tesseract-path "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Important options are:

| Option | Values / default | Meaning |
| --- | --- | --- |
| `--component-solver` | `sequential` or `global`; default `sequential` | Selects component grouping. |
| `--contact-solver` | `off`, `risk_averse`, or `missing_edge`; default `off` | Selects contact-graph inference. |
| `--disable-ocr` | off by default | Skips OCR completely and does not invoke Tesseract. |
| `--no-groundplane` | off by default | Disables copper-plane detection. |
| `--filter-combinations` | `15` | Sets the trace-mask voting search size. |
| `--alignment-threshold` | `0.5` | Sets the minimum accepted alignment score. |
| `--output-folder` | timestamped folder | Selects an explicit report destination. |

## Ablation Evaluation

Run all supported configurations on all samples:

```bash
python evaluation/run_ablations.py
```

This executes 72 pipeline runs: six configurations for 12 samples. It can take
considerably longer than a baseline evaluation. A smaller check can select one
sample and one or more configurations:

```bash
python evaluation/run_ablations.py \
  --sample Simple \
  --configuration baseline \
  --configuration no_ocr \
  --allow-fail
```

PowerShell uses backticks for line continuation:

```powershell
python evaluation/run_ablations.py `
  --sample Simple `
  --configuration baseline `
  --configuration no_ocr `
  --allow-fail
```

The supported configurations are:

| Configuration | Difference from the baseline |
| --- | --- |
| `baseline` | Sequential component solver, default contact graph, OCR, and groundplane handling. |
| `global_component_solver` | Enables global component candidate selection. |
| `contact_risk_averse` | Enables removal of high-risk contact edges. |
| `contact_missing_edge` | Also permits positive missing-edge hypotheses. |
| `no_ocr` | Skips OCR before Tesseract is called. |
| `no_groundplane` | Disables copper-plane detection. |

`no_via_pairing`, `hough_only_pads`, and `no_relaxed_contact` are listed in
the generated manifest as unsupported because the current public API has no
independent switches for them.

OCR has a deliberately narrow role in the current pipeline: it recognizes
printed labels and uses them to infer component identity. Disabling OCR is
therefore expected primarily to reduce recognized text and component
assignments. It is not expected to directly alter pad, trace, or net
reconstruction metrics. If no text is recognized, the OCR contribution is
zero; this is why OCR statistics are also reported separately.

## Generated Reports

A single evaluation creates:

```text
evaluation/results/<timestamp>/
|-- evaluation_report.json     Complete machine-readable results
|-- evaluation_report.md       Human-readable per-sample table
`-- runs/                      Full PRISM output for every sample
```

An ablation run creates:

```text
evaluation/results/ablations/<timestamp>/
|-- ablation_manifest.json
|-- ablation_summary.json
|-- ablation_summary.md
|-- ablation_summary_table.tex
|-- baseline/
|-- global_component_solver/
|-- contact_risk_averse/
|-- contact_missing_edge/
|-- no_ocr/
`-- no_groundplane/
```

All report schemas, keys, values written by the evaluation tools, filenames,
command-line messages, and Markdown/LaTeX labels are in English.

## Checking the Results

Use the following order:

1. Check the process exit code. Exit code 0 means that all selected quality
   thresholds passed, unless `--allow-fail` was used.
2. Open `evaluation_report.md` or `ablation_summary.md` for the compact table.
3. Inspect the `summary` object in the corresponding JSON report. The
   `quality_status`, `pipeline_errors`, and `quality_failures` fields identify
   the samples requiring attention.
4. Inspect each sample's `quality.failures` array to see the exact threshold,
   observed value, and required value.
5. Open the sample's `runs/.../work/quality_report.html` and diagnostic JSON
   files to investigate geometry, component, or connectivity failures.
6. Use the generated `.kicad_sch`, `.layout.json`, and `.roundtrip.json` files
   to inspect schematic export, placement, routing, and KiCad validation.

The main reported groups are:

- pad precision and recall;
- paired-hole precision and recall;
- reconstructed net-size signature precision and recall;
- component reference precision and recall;
- pin-connection true positives, false positives, and false negatives;
- netlist edit distance;
- open diagnostic problem count;
- symbol collisions and KiCad round-trip status;
- OCR text count and OCR availability.

Quality thresholds are versioned in [`metrics.py`](metrics.py). They are
regression limits for the controlled dataset, not estimates of real-world
generalization.

## Summarizing Existing Reports

Compare independently generated reports without rerunning the pipeline:

```bash
python evaluation/summarize_reports.py \
  path/to/first/evaluation_report.json \
  path/to/second/evaluation_report.json \
  --output comparison.json \
  --markdown comparison.md \
  --latex comparison_table.tex
```

This command reads existing reports only and generates an English JSON summary,
Markdown table, and publication-ready LaTeX table.

