#!/usr/bin/env python3
"""Prepare, but never commit automatically, the Step 5B holdout-opening record.

This is a provenance validator. It does not touch EEG and it does not modify the
repository. It emits a candidate HOLDOUT_GATE.json only after the final
numerical profile and the complete 34-subject development sign-off are present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--development-summary",type=Path,required=True)
    ap.add_argument("--development-signoff",type=Path,required=True)
    ap.add_argument("--development-run-id",required=True)
    ap.add_argument("--aggregate-artifact-id",required=True)
    ap.add_argument("--aggregate-artifact-digest",required=True)
    ap.add_argument("--opened-at-utc",required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    here=Path(__file__).parent
    profile=json.loads((here/"NUMERICAL_PROFILE.json").read_text())
    current_gate=json.loads((here/"HOLDOUT_GATE.json").read_text())

    if profile.get("status")!="FROZEN":
        raise RuntimeError("Holdout cannot open before the final numerical profile is FROZEN")
    if not profile.get("final_freeze_commit"):
        raise RuntimeError("Frozen numerical profile lacks final_freeze_commit")
    if current_gate.get("status")!="CLOSED":
        raise RuntimeError("Refusing to prepare a second holdout opening")

    summary=json.loads(args.development_summary.read_text())
    if summary.get("phase")!="development":
        raise RuntimeError("Wrong development summary")
    if summary.get("confirmatory_inference_performed") is not False:
        raise RuntimeError("Development summary contains confirmatory inference")
    if summary.get("holdout_subjects_seen") is not False:
        raise RuntimeError("Development summary reports holdout signal access")
    if int(summary.get("n_subjects",0))!=34:
        raise RuntimeError("Final development summary must contain exactly 34 subjects")
    if int(summary.get("optimizer_calls",0))!=136:
        raise RuntimeError("Final development summary must contain 136 optimizer calls")
    if summary.get("all_optimizers_success") is not True:
        raise RuntimeError("Development optimizer audit did not pass")
    if summary.get("numerical_profile",{}).get("final_freeze_commit") != profile["final_freeze_commit"]:
        raise RuntimeError("Development aggregate was not generated under the final frozen profile")

    signoff=args.development_signoff.read_text(encoding="utf-8")
    if "PASS - HOLDOUT MAY OPEN" not in signoff:
        raise RuntimeError("Development sign-off has not explicitly passed")

    digest=args.aggregate_artifact_digest
    if not (digest.startswith("sha256:") and len(digest)==71):
        raise RuntimeError("Aggregate artifact digest must be canonical sha256:<64hex>")
    int(digest.split(":",1)[1],16)

    gate={
        "status":"OPEN",
        "opened_at_utc":args.opened_at_utc,
        "development_run_id":str(args.development_run_id),
        "development_aggregate_artifact_id":str(args.aggregate_artifact_id),
        "development_aggregate_sha256":digest,
        "development_summary_file_sha256":"sha256:"+sha256(args.development_summary),
        "development_signoff_file_sha256":"sha256:"+sha256(args.development_signoff),
        "numerical_freeze_commit":profile["final_freeze_commit"],
        "numerical_profile_sha256":"sha256:"+sha256(here/"NUMERICAL_PROFILE.json"),
        "required_development_subjects":34,
        "holdout_first_subject":"sub-044",
        "holdout_last_subject":"sub-608",
        "holdout_subject_count":565,
        "primary_condition":"ses-1 / EyesClosed / acq-pre",
        "note":(
            "Provenance-only opening record prepared after the complete final development "
            "aggregate and sign-off. No scientific setting is changed by this file."
        ),
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(gate,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(gate,indent=2))


if __name__=="__main__":
    main()
