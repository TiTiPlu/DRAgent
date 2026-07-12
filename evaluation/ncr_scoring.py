"""
Guideline-compliance scoring (Methods 4.3.1).

Each plan is scored by a third-party LLM against the criteria in Supplementary
Tables 1 (ophthalmic) and 2 (endocrine). The scorer is hosted as a Dify workflow
that wraps the scoring prompt (Supplementary Note 3) and returns one TRUE/FALSE
verdict per criterion.

A dynamic triggering mechanism means only the criteria relevant to a case are
returned, so the denominator varies by case. The Normalized Compliance Rate is

    NCR = actual score / maximum score of triggered items

Usage:
    python ncr_scoring.py --scheme ophthalmic --layout agent \\
        --run-root run/ --report-root reports/ --out ncr_ophthalmic.csv
"""

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests

from loader import iter_cases, read

# Scoring criteria. Item weights are the clinical importance weights agreed by
# two experienced clinicians; the full item text is in Supplementary Tables 1-2.
SCHEMES = {
    # Ophthalmic: normativity (S), correctness (C), procedure (P), decision (D)
    "ophthalmic": {
        "weights": {
            "S1": 0.5, "S2": 0.5, "S3": 0.5, "S4": 0.5, "S5": 0.5, "S6": 0.5, "S7": 0.5,
            "C1": 3, "C2": 3, "C3": 3, "C4": 3, "C5": 3, "C6": 3,
            "P1": 3, "P2": 1, "P3": 1, "P4": 1, "P5": 3, "P6": 3, "P7": 3, "P8": 2,
            "D1": 1, "D2": 3, "D3": 3,
        },
        # Items that must always be returned; a response missing them is invalid.
        "always_required": {"S1", "S2", "S3", "S4", "S5", "S6", "S7"},
        # Items additionally required once the case is severe enough for surgery.
        "required_from_grade_3": {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"},
        "env_key": "DIFY_SCORER_OPHTHALMIC_KEY",
    },
    # Endocrine: normativity (A), correctness (X), procedure (U), follow-up (F)
    "endocrine": {
        "weights": {
            "A1": 1, "A2": 1,
            "X1": 3, "X2": 3, "X3": 3, "X4": 3,
            "U1": 1, "U2": 1,
            "F1": 1, "F2": 1,
        },
        "always_required": set(),
        "required_from_grade_3": set(),
        "env_key": "DIFY_SCORER_ENDOCRINE_KEY",
    },
}

DIFY_WORKFLOW_URL = os.environ.get("DIFY_WORKFLOW_URL", "https://api.dify.ai/v1/workflows/run")
MAX_RETRIES = 3
WORKERS = 6


def parse_verdicts(text: str, weights: dict) -> dict:
    """Pull `<item id> TRUE|FALSE` pairs out of the scorer's reply."""
    verdicts = {}
    for line in text.splitlines():
        m = re.search(r"([A-Z]\d+)[\s:：]*([^\s:：]+)", line.strip())
        if not m:
            continue
        item = m.group(1).upper()
        raw = m.group(2).upper().replace("*", "").strip()
        status = re.sub(r"[^A-Z].*", "", raw)
        if item in weights:
            verdicts[item] = (status == "TRUE")
    return verdicts


def _parse_workflow_sse(response):
    for raw in response.iter_lines():
        if not raw:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not raw.startswith("data:"):
            continue
        payload = raw[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "workflow_finished":
            data = event.get("data", {})
            if data.get("status") != "succeeded":
                return None
            outputs = data.get("outputs", {})
            if isinstance(outputs, str):
                try:
                    outputs = json.loads(outputs)
                except json.JSONDecodeError:
                    outputs = {"text": outputs}
            return outputs
        if event.get("event") == "error":
            return None
    return None


def call_scorer(api_key: str, plan: str, report: str, grade: str):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": {"treatment": plan, "report": report, "type": grade},
        "response_mode": "streaming",
        "user": "ncr-scorer",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(DIFY_WORKFLOW_URL, headers=headers, json=payload,
                                 stream=True, timeout=180)
            resp.raise_for_status()
            outputs = _parse_workflow_sse(resp)
            if outputs:
                return outputs
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"scorer failed: {e}", flush=True)
    return None


def score_case(case, scheme, api_key, writer, lock):
    cfg = SCHEMES[scheme]
    weights = cfg["weights"]

    outputs = call_scorer(api_key, read(case.plan_path), read(case.report_path), case.grade)
    if not outputs:
        return None

    text = outputs.get("text") or json.dumps(outputs, ensure_ascii=False)
    verdicts = parse_verdicts(text, weights)

    # Reject incomplete responses: the always-required items must all be present,
    # and from DR-3 onward the procedure items must be present too.
    missing = cfg["always_required"] - set(verdicts)
    if missing:
        print(f"{case.case_id}: incomplete response, missing {sorted(missing)}", flush=True)
        return None
    grade_num = int(re.search(r"(\d)", case.grade).group(1))
    if grade_num >= 3:
        missing = cfg["required_from_grade_3"] - set(verdicts)
        if missing:
            print(f"{case.case_id}: incomplete response, missing {sorted(missing)}", flush=True)
            return None

    triggered_max = sum(weights[i] for i in verdicts)
    actual = sum(weights[i] for i, ok in verdicts.items() if ok)
    ncr = actual / triggered_max if triggered_max else 0.0

    row = {
        "model": case.model, "grade": case.grade, "case_id": case.case_id,
        "actual_score": actual, "triggered_max": triggered_max, "ncr": f"{ncr:.4f}",
        "triggered_items": ",".join(sorted(verdicts)),
    }
    for item in weights:
        row[item] = ("1" if verdicts[item] else "0") if item in verdicts else "/"

    with lock:
        writer.writerow(row)
    return ncr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=SCHEMES, required=True)
    ap.add_argument("--layout", choices=["agent", "llm"], required=True,
                    help="'llm' also covers HuatuoGPT-3")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--report-root", required=True)
    ap.add_argument("--stage", default="final", help="DRAgent stage to score")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = SCHEMES[args.scheme]
    api_key = os.environ[cfg["env_key"]]

    cases = list(iter_cases(args.run_root, args.report_root, args.layout,
                            args.stage, args.models))
    print(f"{len(cases)} cases", flush=True)

    fields = (["model", "grade", "case_id", "actual_score", "triggered_max", "ncr",
               "triggered_items"] + list(cfg["weights"]))
    lock = Lock()
    scores = []

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(score_case, c, args.scheme, api_key, writer, lock)
                       for c in cases]
            for fut in as_completed(futures):
                ncr = fut.result()
                if ncr is not None:
                    scores.append(ncr)

    if scores:
        print(f"mean NCR: {100 * sum(scores) / len(scores):.2f}%  (n={len(scores)})", flush=True)


if __name__ == "__main__":
    main()
