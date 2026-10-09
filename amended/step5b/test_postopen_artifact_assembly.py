#!/usr/bin/env python3
"""Model-blind, entirely synthetic end-to-end test of the post-open artefact assembler.

Never downloads holdout fit outcomes or executes model fitting or inference.
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/step5b-postopen-final-aggregate-only.yml")
FROZEN_FIT_BLOB = "f2276f20ff6eb20666634b2b20fa21b66288780d"
B = {f"sub-{i:03d}" for i in range(232, 420)}
AC = ({f"sub-{i:03d}" for i in range(44, 232) if i not in (75, 206, 230)}
      | {f"sub-{i:03d}" for i in range(420, 609) if i != 425})
S = {"sub-206", "sub-230", "sub-425"}
assert len(B) == 188 and len(AC) == 373
assert B | AC | {"sub-075"} | S == {f"sub-{i:03d}" for i in range(44, 609)}


def assembler_python() -> str:
    data = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    steps = data["jobs"]["aggregate"]["steps"]
    stages = [step["run"] for step in steps
              if step.get("name", "").startswith("Assemble 565 assignments")]
    assert len(stages) == 1
    lines = stages[0].splitlines()
    assert lines[0] == "python - <<'PY'" and lines[-1] == "PY"
    source = "\n".join(lines[1:-1])
    compile(source, str(WORKFLOW), "exec")
    return source


def write_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data) + "\n", encoding="utf-8")


def fixture(root: Path) -> None:
    for folder, subjects in (("src_b", B), ("src_ac", AC), ("src_075", {"sub-075"})):
        for s in sorted(subjects):
            name = s + (".excluded.json" if s in ("sub-233", "sub-045") else ".json")
            write_json(root / folder / name, {
                "subject": s, "synthetic_only": True,
                "real_fit_scores": "NOT_IN_THIS_TEST",
            })
            if folder == "src_b":
                continue
            a1 = s == "sub-075"
            write_json(root / folder / ("source_" + s + ".json"), {
                "subject": s,
                "analysis_classification": "POST_OPEN_AMENDED",
                "original_scientific_fit_blob": FROZEN_FIT_BLOB,
                "source_type": "A1_AMENDED_SUB075" if a1 else "ORIGINAL_FROZEN_PREPROCESS",
                "preprocessing_run_id": 37786081008 if a1 else 37752719639,
            })
    for s in S:
        write_json(root / "amended/step5b/ineligible" / (s + ".ineligible.json"),
                   {"subject": s, "synthetic_only": True})


def execute(code: str, mutate=None, should_pass=True, label="baseline") -> None:
    with tempfile.TemporaryDirectory(prefix="smm_postopen_synthetic_") as tmp:
        root = Path(tmp)
        fixture(root)
        if mutate is not None:
            mutate(root)
        proc = subprocess.run([sys.executable, "-c", code], cwd=root,
                              capture_output=True, text=True, timeout=30)
        if should_pass:
            assert proc.returncode == 0, f"{label} unexpectedly failed: {proc.stderr[-1600:]}"
            out = root / "amended_inputs"
            assert out.is_dir()
            manifest = json.loads((out / "artifact_sources.json").read_text())
            assert manifest["analysis_classification"] == "POST_OPEN_AMENDED"
            assert len(manifest["sourced_fits"]) == 562
            assert manifest["sourced_fits"]["sub-075"] == "amended_A_C"
            assert all(manifest["sourced_fits"][s] == "original_B" for s in B)
            assert len([x for x in out.iterdir() if x.name.startswith("sub-")]) == 565
            assert len([x for x in out.iterdir() if x.name.startswith("source_")]) == 374
            assert (out / "sub-233.excluded.json").exists()
            assert (out / "sub-045.excluded.json").exists()
            print(f"SYNTHETIC_ASSEMBLY_PASS {label}")
        else:
            assert proc.returncode != 0, f"{label} should fail closed"
            print(f"SYNTHETIC_ASSEMBLY_REJECTION_PASS {label}")


def main():
    code = assembler_python()
    execute(code, label="complete_with_two_frozen_QC_exclusions")
    execute(code, lambda r: (r / "src_b/sub-419.json").unlink(),
            should_pass=False, label="missing_original_B")
    execute(code, lambda r: (r / "src_ac/source_sub-044.json").unlink(),
            should_pass=False, label="missing_amended_source_manifest")
    execute(code, lambda r: write_json(r / "src_b/sub-232.excluded.json",
                                        {"synthetic_only": True}),
            should_pass=False, label="duplicate_B_fit_and_exclusion")
    execute(code, lambda r: write_json(r / "src_ac/sub-075.json",
                                        {"synthetic_only": True}),
            should_pass=False, label="cross_run_075_collision")
    execute(code, lambda r: write_json(r / "src_075/source_sub-075.json", {
        "subject": "sub-075",
        "analysis_classification": "POST_OPEN_AMENDED",
        "original_scientific_fit_blob": FROZEN_FIT_BLOB,
        "source_type": "ORIGINAL_FROZEN_PREPROCESS",
        "preprocessing_run_id": 37752719639,
    }), should_pass=False, label="wrong_075_preprocessing_provenance")
    execute(code, lambda r: (r / "src_b/unexpected.txt").write_text("rogue"),
            should_pass=False, label="unexpected_artefact")
    print("MODEL_BLIND_ASSEMBLY_TEST_SUITE_PASS")


if __name__ == "__main__":
    main()
