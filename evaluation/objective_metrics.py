"""
Objective similarity between generated and reference treatment plans
(Methods 4.3.2).

Both the generated plan and the physician reference are first distilled into the
same decision fields (distill.py). Similarity is computed per field and
aggregated by a weighted sum, with weights taken from the expert-determined
clinical importance of each field (Supplementary Table 3): the surgical decision
and the anti-VEGF decision carry the most weight, individual laser parameters the
least, and fields that vary little across cases (patient education, precautions)
are excluded from the aggregate.

Metrics: BLEU-1..4, ROUGE-L, METEOR, BERTScore-F1.

Usage:
    python objective_metrics.py --layout agent --run-root run/ \\
        --report-root reports/ --ref-root reference/ --out metrics_agent.csv
"""

import argparse
import os
import re
import warnings

import jieba
import pandas as pd
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_chinese import Rouge

from loader import iter_cases, distilled_path, read, resolve_reference_path

warnings.filterwarnings("ignore")

# Fields produced by distill.py.
FIELDS = [
    "疾病名称", "病情描述", "治疗原则",
    "当前是否需要进行激光手术", "手术类型", "激光类型", "光斑大小", "功率",
    "曝光时间", "治疗分期", "避让区域", "激光斑总数", "其他手术",
    "当前是否使用抗VEGF", "抗VEGF使用方法", "其他药物",
    "血糖/血压控制方法", "随访频率", "随访项目",
    "患者教育", "注意事项",
]

# Aggregation weights (Supplementary Table 3). Normalised to sum to 1 below.
WEIGHTS = {
    "当前是否需要进行激光手术": 0.2, "手术类型": 0.1,
    "激光类型": 0.025, "光斑大小": 0.025, "功率": 0.025, "曝光时间": 0.025,
    "治疗分期": 0.025, "避让区域": 0.025, "激光斑总数": 0.025, "其他手术": 0.1,
    "当前是否使用抗VEGF": 0.2, "抗VEGF使用方法": 0.05, "其他药物": 0.05,
    "血糖/血压控制方法": 0.025, "随访频率": 0.05, "随访项目": 0.05,
}
_total = sum(WEIGHTS.values())
WEIGHTS = {k: v / _total for k, v in WEIGHTS.items()}

METRICS = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", "rouge-l", "meteor", "bertscore-f1"]

ALIASES = {
    "二. 治疗原则": "治疗原则",
    "五. 血糖/血压控制方法": "血糖/血压控制方法",
    "七. 患者教育": "患者教育",
    "八. 注意事项": "注意事项",
}


def tokenize(text: str) -> str:
    return " ".join(jieba.cut(text.strip()))


def parse_fields(text: str) -> dict:
    """Split a distilled plan into its fields."""
    out = {f: "" for f in FIELDS}
    current = None
    for line in text.split("\n"):
        line = line.rstrip()
        m = re.match(r"^\s*([^：]+)：\s*(.*)$", line)
        if m:
            name = ALIASES.get(m.group(1).strip(), m.group(1).strip())
            if name in out:
                current = name
                if m.group(2).strip():
                    out[name] = (out[name] + " " + m.group(2).strip()).strip()
            else:
                current = None
            continue
        if re.match(r"^[一二三四五六七八]\.", line):
            current = None
            continue
        if current and line:
            out[current] = (out[current] + " " + line.strip()).strip()
    return out


def bleu_scores(gen: str, ref: str) -> dict:
    if not gen.strip() or not ref.strip():
        return {f"bleu-{n}": 0.0 for n in range(1, 5)}
    g, r = tokenize(gen).split(), tokenize(ref).split()
    sm = SmoothingFunction().method1
    ws = {1: (1, 0, 0, 0), 2: (0.5, 0.5, 0, 0),
          3: (1/3, 1/3, 1/3, 0), 4: (0.25, 0.25, 0.25, 0.25)}
    try:
        return {f"bleu-{n}": sentence_bleu([r], g, weights=ws[n], smoothing_function=sm)
                for n in range(1, 5)}
    except Exception:
        return {f"bleu-{n}": 0.0 for n in range(1, 5)}


def rouge_l(gen: str, ref: str) -> float:
    if not gen.strip() or not ref.strip():
        return 0.0
    try:
        return Rouge().get_scores(tokenize(gen), tokenize(ref))[0]["rouge-l"]["f"]
    except Exception:
        return 0.0


def meteor(gen: str, ref: str) -> float:
    if not gen.strip() or not ref.strip():
        return 0.0
    try:
        return meteor_score([tokenize(ref).split()], tokenize(gen).split())
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["agent", "llm"], required=True,
                    help="'llm' also covers HuatuoGPT-3")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--report-root", default=None,
                    help="required for LLM; optional explicit report source for a consolidated agent tree")
    ap.add_argument("--ref-root", required=True,
                    help="distilled physician reference plans, one per case_id")
    ap.add_argument("--stage", default="final")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--model-name", default=None,
                    help="label for an agent pipeline run (default: run-root basename)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = list(iter_cases(args.run_root, args.report_root, args.layout,
                            args.stage, args.models, args.model_name))
    print(f"{len(cases)} cases", flush=True)
    if not cases:
        raise SystemExit(
            "no cases found; check the selected layout and roots. A pipeline "
            "agent tree keeps reports beside plans; a consolidated agent tree "
            "and an LLM tree require --report-root"
        )

    scorer = BERTScorer(model_type="bert-base-chinese", lang="zh")

    rows = []
    for case in cases:
        gen_path = distilled_path(case, args.layout)
        ref_path = resolve_reference_path(args.ref_root, case.grade, case.case_id)
        if not os.path.exists(gen_path) or not os.path.exists(ref_path):
            continue

        gen = parse_fields(read(gen_path))
        ref = parse_fields(read(ref_path))

        # BERTScore is batched across all weighted fields of this case.
        pairs = [(f, gen[f], ref[f]) for f in WEIGHTS
                 if gen[f].strip() and ref[f].strip()]
        bert = {}
        if pairs:
            _, _, f1 = scorer.score([p[1] for p in pairs], [p[2] for p in pairs])
            bert = {p[0]: float(v) for p, v in zip(pairs, f1)}

        weighted = {m: 0.0 for m in METRICS}
        for field, w in WEIGHTS.items():
            g, r = gen[field], ref[field]
            s = bleu_scores(g, r)
            s["rouge-l"] = rouge_l(g, r)
            s["meteor"] = meteor(g, r)
            s["bertscore-f1"] = bert.get(field, 0.0)
            for m in METRICS:
                weighted[m] += w * s[m]

        rows.append({"model": case.model, "grade": case.grade, "case_id": case.case_id,
                     **{m: round(weighted[m], 4) for m in METRICS}})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    if df.empty:
        raise SystemExit(
            "no cases were scored; check the run/report/reference roots, layout, "
            "stage, and required distilled files"
        )
    print(df.groupby("model")[METRICS].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
