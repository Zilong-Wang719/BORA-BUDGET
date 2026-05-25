from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


FILES = {
    17: {
        "no_think": "artifacts/remote_stage_c/math500_switch_20260512/standard_direct_cot_math500_seed17.json",
        "think8k": "artifacts/remote_stage_c/math500_think8k_baseline_20260512/standard_direct_cot_think8k_math500_seed17_merged.json",
        "think12k": "artifacts/remote_stage_c/math500_think12k_baseline_20260512/standard_direct_cot_think12k_math500_seed17_merged.json",
        "sc3": "artifacts/remote_stage_c/math500_sc3_seed17_20260515/self_consistency3_no_think_math500_seed17.json",
    },
    7: {
        "no_think": "artifacts/remote_stage_c/math500_seed_repeat_seed7_20260513/standard_direct_cot_no_think_math500_seed7.json",
        "think8k": "artifacts/remote_stage_c/math500_think8k_seed7_20260515/standard_direct_cot_think8k_math500_seed7.json",
        "think12k": "artifacts/remote_stage_c/math500_think12k_seed7_20260515/standard_direct_cot_think12k_math500_seed7.json",
        "sc3": "artifacts/remote_stage_c/math500_sc3_seed7_20260515/self_consistency3_no_think_math500_seed7.json",
    },
    23: {
        "no_think": "artifacts/remote_stage_c/math500_seed_repeat_seed23_20260515/standard_direct_cot_no_think_math500_seed23.json",
        "think8k": "artifacts/remote_stage_c/math500_think8k_seed23_20260515/standard_direct_cot_think8k_math500_seed23.json",
        "think12k": "artifacts/remote_stage_c/math500_think12k_seed23_20260515/standard_direct_cot_think12k_math500_seed23.json",
        "sc3": "artifacts/remote_stage_c/math500_sc3_seed23_20260515/self_consistency3_no_think_math500_seed23.json",
    },
}


def run_eval(path: Path) -> dict:
    summary_path = Path(str(path) + ".robust_summary.json")
    jsonl_path = Path(str(path) + ".robust_eval.jsonl")
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_math_answers_robust.py",
            "--input",
            str(path),
            "--output-summary",
            str(summary_path),
            "--output-jsonl",
            str(jsonl_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for seed, mapping in FILES.items():
        for kind, rel in mapping.items():
            path = Path(rel)
            if not path.exists():
                print(f"MISSING seed={seed} kind={kind} path={path}")
                continue
            summary = run_eval(path)
            rows.append(
                {
                    "seed": seed,
                    "kind": kind,
                    "count": summary["count"],
                    "old_correct": summary["old_correct"],
                    "old_accuracy": summary["old_accuracy"],
                    "robust_correct": summary["robust_correct"],
                    "robust_accuracy": summary["robust_accuracy"],
                    "old_to_new_correct": summary["old_to_new_correct"],
                    "old_correct_to_new_wrong": summary["old_correct_to_new_wrong"],
                    "match_methods": summary["match_methods"],
                }
            )
            print(
                f"seed={seed} kind={kind} "
                f"old={summary['old_correct']}/{summary['count']} ({100*summary['old_accuracy']:.2f}%) "
                f"robust={summary['robust_correct']}/{summary['count']} ({100*summary['robust_accuracy']:.2f}%) "
                f"+{summary['old_to_new_correct']} -{summary['old_correct_to_new_wrong']}"
            )
    out = Path("artifacts/eval_audit/math500_robust_recheck_20260520.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
