"""
Locate treatment plans and reference clinical records on disk.

Two output layouts exist:

  DRAgent            run/{model}/{DR-N}/{case_id}/{case_id}_{stage}.md
  standalone LLM     run/{model}/{DR-N}/{case_id}.md
  (HuatuoGPT-3 uses the standalone-LLM layout)

Reference records live in a flat tree:

  reports/{case_id}/眼底病病例报告.md

Every evaluation script goes through iter_cases() so the layout difference
never leaks into the scoring or metric code.
"""

import os
from pathlib import Path
from typing import Iterator, NamedTuple

DR_GRADES = ["DR-1", "DR-2", "DR-3", "DR-4"]
REPORT_FILENAME = "眼底病病例报告.md"


class Case(NamedTuple):
    model: str          # backbone directory name
    grade: str          # DR-1 .. DR-4
    case_id: str
    plan_path: str      # generated treatment plan
    report_path: str    # reference clinical record


def iter_cases(run_root: str, report_root: str, layout: str,
               stage: str = "final", models=None) -> Iterator[Case]:
    """Yield one Case per (model, grade, case_id).

    layout: "agent" for the DRAgent tree, "llm" for the flat tree.
    stage:  which DRAgent stage file to score ("initial", "surgical", "final").
            Ignored when layout == "llm".
    models: optional whitelist of backbone directory names.
    """
    if layout not in ("agent", "llm"):
        raise ValueError(f"unknown layout: {layout}")

    for model in sorted(os.listdir(run_root)):
        model_dir = os.path.join(run_root, model)
        if not os.path.isdir(model_dir) or (models and model not in models):
            continue

        for grade in DR_GRADES:
            grade_dir = os.path.join(model_dir, grade)
            if not os.path.isdir(grade_dir):
                continue

            for entry in sorted(os.listdir(grade_dir)):
                entry_path = os.path.join(grade_dir, entry)

                if layout == "agent":
                    if not os.path.isdir(entry_path):
                        continue
                    case_id = entry
                    plan_path = os.path.join(entry_path, f"{case_id}_{stage}.md")
                else:
                    if not entry.endswith(".md") or entry.endswith("_abstract.md"):
                        continue
                    case_id = entry[:-3]
                    plan_path = entry_path

                report_path = os.path.join(report_root, case_id, REPORT_FILENAME)
                if not os.path.exists(plan_path) or not os.path.exists(report_path):
                    continue

                yield Case(model, grade, case_id, plan_path, report_path)


def distilled_path(case: Case, layout: str) -> str:
    """Where the distilled plan for this case lives (see distill.py)."""
    if layout == "agent":
        stem = Path(case.plan_path).stem            # {case_id}_{stage}
        return os.path.join(Path(case.plan_path).parent, f"{stem}_abstract.md")
    return os.path.join(Path(case.plan_path).parent, f"{case.case_id}_abstract.md")


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
