"""
Locate treatment plans and reference clinical records on disk.

Two output layouts exist.  The agent layout is exactly the tree emitted by
``agents/pipeline.py``:

  DRAgent            run/{DR-N}/{case_id}/{case_id}_{stage}.md
                     run/{DR-N}/{case_id}/眼底病病例报告.md
  standalone LLM     run/{model}/{DR-N}/{case_id}.md
  (HuatuoGPT-3 uses the standalone-LLM layout)

Standalone-LLM reference records and physician-reference plans use a
grade-stratified tree:

  reports/{DR-N}/{case_id}/眼底病病例报告.md
  reference/{DR-N}/{case_id}/治疗方案_abstract.md

Every evaluation script goes through iter_cases() so the layout difference
never leaks into the scoring or metric code.
"""

import os
from pathlib import Path
from typing import Iterator, NamedTuple

DR_GRADES = ["DR-1", "DR-2", "DR-3", "DR-4"]
REPORT_FILENAME = "眼底病病例报告.md"
REFERENCE_FILENAME = "治疗方案_abstract.md"


class Case(NamedTuple):
    model: str          # backbone directory name
    grade: str          # DR-1 .. DR-4
    case_id: str
    plan_path: str      # generated treatment plan
    report_path: str    # reference clinical record


def iter_cases(run_root: str, report_root: str | None, layout: str,
               stage: str = "final", models=None,
               model_name: str | None = None) -> Iterator[Case]:
    """Yield one Case per (model, grade, case_id).

    layout: "agent" for the DRAgent tree, "llm" for the flat tree.
    stage:  which DRAgent stage file to score ("initial", "surgical", "final").
            Ignored when layout == "llm".
    models: optional whitelist of standalone-LLM directory names.
    model_name: label for a pipeline run; defaults to the run-root directory name.
    report_root: for agent output, omit it for a complete pipeline tree (the
                 report is beside the plan), or pass it explicitly when scoring
                 a consolidated plan-only tree. Required for standalone LLMs.
    """
    if layout not in ("agent", "llm"):
        raise ValueError(f"unknown layout: {layout}")

    if layout == "agent":
        model_dirs = [(model_name or Path(run_root).resolve().name, run_root)]
    else:
        model_dirs = [
            (model, os.path.join(run_root, model))
            for model in sorted(os.listdir(run_root))
            if os.path.isdir(os.path.join(run_root, model))
            and (not models or model in models)
        ]

    for model, model_dir in model_dirs:

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

                if layout == "agent":
                    if report_root:
                        report_path = os.path.join(
                            report_root, grade, case_id, REPORT_FILENAME)
                    else:
                        report_path = os.path.join(entry_path, REPORT_FILENAME)
                else:
                    if not report_root:
                        raise ValueError("report_root is required for the llm layout")
                    report_path = os.path.join(report_root, grade, case_id, REPORT_FILENAME)
                if not os.path.exists(plan_path) or not os.path.exists(report_path):
                    continue

                yield Case(model, grade, case_id, plan_path, report_path)


def resolve_reference_path(ref_root: str, grade: str, case_id: str) -> str:
    """Return the canonical grade-stratified physician-reference path."""
    return os.path.join(ref_root, grade, case_id, REFERENCE_FILENAME)


def distilled_path(case: Case, layout: str) -> str:
    """Where the distilled plan for this case lives (see distill.py)."""
    if layout == "agent":
        stem = Path(case.plan_path).stem            # {case_id}_{stage}
        return os.path.join(Path(case.plan_path).parent, f"{stem}_abstract.md")
    return os.path.join(Path(case.plan_path).parent, f"{case.case_id}_abstract.md")


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
