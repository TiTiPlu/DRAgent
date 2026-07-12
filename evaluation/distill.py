"""
Structured distillation of treatment plans (Methods 4.3.2).

Surface-level token matching cannot separate two plans that differ in a single
decisive word ("perform surgery" vs "do not perform surgery"). Before computing
objective similarity, every plan is therefore distilled into the same set of
decision fields, and similarity is computed field by field.

The same prompt is applied to every system under evaluation - DRAgent, each
standalone LLM, HuatuoGPT-3, and the physician reference - so that no system is
advantaged by its preprocessing.

Usage:
    python distill.py --layout agent --run-root run/ --report-root reports/
    python distill.py --layout llm   --run-root run/ --report-root reports/
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from loader import iter_cases, distilled_path, read

DISTILL_URL = os.environ.get("DISTILL_URL", "https://api.deepseek.com/v1/chat/completions")
DISTILL_MODEL = os.environ.get("DISTILL_MODEL", "deepseek-chat")
MAX_RETRIES = 3
WORKERS = 32

SYSTEM_PROMPT = "你的责任是对输入文本进行格式化，确保只提取关键信息，务必简洁明了！务必完全遵循模版！"

TEMPLATE = """请根据提供给你的治疗方案内容({content})，按照指定的模板只提取每个部分的关键信息，务必确保内容简洁明了！整理只针对当下治疗，不包含假设情况部分。务必完全遵循模版，若未出现请不要自行省略，在该项后面直接写"无"！以下是模板，生成时去除括号内容：
一. 疾病诊断（简要）
    疾病名称：
    病情描述：
二. 治疗原则：
三. 治疗方案（未提及写无）
    当前是否需要进行激光手术：是/否(禁止出现假设性表达)
    手术类型：
    激光类型：
    光斑大小：
    功率：
    曝光时间：
    治疗分期：
    避让区域：
    激光斑总数：
    其他手术：
四. 药物使用（未提及写无）
    当前是否使用抗VEGF：是/否(禁止出现假设性表达)
    抗VEGF使用方法：
    其他药物：
五. 血糖/血压控制方法：
六. 随访计划
    随访频率：
    随访项目：
七. 患者教育：
八. 注意事项：
生成时去除括号及其中内容"""


def distil(in_path: str, out_path: str, overwrite: bool = False) -> bool:
    if os.path.exists(out_path) and not overwrite:
        return True

    payload = {
        "model": DISTILL_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TEMPLATE.format(content=read(in_path))},
        ],
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {os.environ['DISTILL_API_KEY']}",
               "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(DISTILL_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(resp.json()["choices"][0]["message"]["content"])
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"failed: {os.path.basename(in_path)}: {e}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["agent", "llm"], required=True,
                    help="'llm' also covers HuatuoGPT-3 and the physician reference")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--report-root", required=True)
    ap.add_argument("--stage", default="final")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cases = list(iter_cases(args.run_root, args.report_root, args.layout,
                            args.stage, args.models))
    print(f"{len(cases)} plans to distil", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(distil, c.plan_path, distilled_path(c, args.layout),
                               args.overwrite) for c in cases]
        for fut in as_completed(futures):
            done += bool(fut.result())

    print(f"{done}/{len(cases)} distilled", flush=True)


if __name__ == "__main__":
    main()
