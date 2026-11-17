# Evaluation Reference Data

This directory contains versioned reference annotations for the 12 synthetic
PCB samples in [`Dataset/`](../../Dataset/). The annotations were derived from
the corresponding KiCad projects and are consumed by
[`run_evaluation.py`](../run_evaluation.py).

Each sample directory contains:

- `expected_stats.json` — reference object and connectivity counts;
- `expected_geometry.json` — pad, via, trace, zone, and outline geometry;
- `expected_components.json` — PCB component and pad assignments;
- `expected_schematic_components.json` — reference schematic components;
- `expected_netlist.json` — reference nets and connected pins.

The TOP and BOTTOM images are not duplicated here. Evaluation inputs are read
directly from `Dataset/<Sample>/TOP.png` and `Dataset/<Sample>/BOTTOM.png`.
The `source_schematic` fields use repository-relative English paths so reports
do not depend on a developer's local filesystem.

These files describe controlled KiCad-derived examples. They are suitable for
repeatable regression measurements but do not constitute annotations for
photographs of physical PCBs.

