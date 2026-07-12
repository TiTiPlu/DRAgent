"""
DRAgent: fundus image -> clinical record -> multidisciplinary treatment plan.

Stages 1-3: generate a clinical record from a fundus image, apply the
            terminology-library style transfer, and export a per-grade CSV.
Stages 4-5: run the multi-agent consultation (ophthalmologist, laser surgeon,
            endocrinologist, pharmacist, nurse) and distil the final plan
            into the treatment-plan template.

API keys are read from environment variables. See README.
"""

import os
import re
import time
import json
import base64
import shutil
import random
import hashlib
import requests
import pandas as pd
from pathlib import Path
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------- Config ---------------------------------
RUN_STEPS_1_TO_3 = False   # image -> clinical record -> style transfer -> CSV
RUN_STEPS_4_TO_5 = True    # multi-agent consultation -> template distillation

OVERWRITE_ALL = False      # False: resume from cached stage files; True: rerun all
SAMPLE_LIMIT = None        # int: cap images per DR grade (steps 1-3 only); None: all

SELECTED_DR_TYPES = ["DR-1", "DR-2", "DR-3", "DR-4"]

IMAGE_ROOTS = [
    os.environ.get("DR_DATA_ROOT", "./data"),
]
FFA_CAPTIONS_JSON = os.environ.get("FFA_CAPTIONS_JSON", "./data/captions.json")

# Backbone LLM for clinical-record generation (outpatient physician agent).
# Swap REPORT_MODEL to reproduce the other backbones reported in the paper.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ["LLM_API_KEY"]
REPORT_MODEL = os.environ.get("REPORT_MODEL", "gpt-5")

# Dify-hosted agents. Each key is a Dify app key for the corresponding DSL
# in agents/dsl/.
DIFY_CHAT_URL = os.environ.get("DIFY_CHAT_URL", "https://api.dify.ai/v1/chat-messages")
DIFY_WORKFLOW_URL = os.environ.get("DIFY_WORKFLOW_URL", "https://api.dify.ai/v1/workflows/run")

STYLE_TRANSFER_KEY = os.environ["DIFY_STYLE_TRANSFER_KEY"]   # style_transfer.yml
EYE_EXPERT_KEY = os.environ["DIFY_OPHTHALMOLOGIST_KEY"]      # ophthalmologist.yml
LASER_EXPERT_KEY = os.environ["DIFY_LASER_SURGEON_KEY"]      # laser_surgeon.yml
ENDO_EXPERT_KEY = os.environ["DIFY_ENDOCRINOLOGIST_KEY"]     # endocrinologist.yml
MED_EXPERT_KEY = os.environ["DIFY_PHARMACIST_KEY"]           # pharmacist.yml
INTEG_KEY = os.environ["DIFY_NURSE_KEY"]                     # nurse.yml

# Plan distillation (Methods 4.3.2)
DISTILL_URL = os.environ.get("DISTILL_URL", "https://api.deepseek.com/v1/chat/completions")
DISTILL_API_KEY = os.environ["DISTILL_API_KEY"]
DISTILL_MODEL = os.environ.get("DISTILL_MODEL", "deepseek-chat")

MAX_RETRIES = 5
API_CALL_DELAY = 1
MAX_ROUNDS = 3             # discussion rounds per specialist (Methods 4.2.3)
WORKERS = 8
# ---------------------------------------------------------------------------

HEADERS_EYE = {"Authorization": f"Bearer {EYE_EXPERT_KEY}", "Content-Type": "application/json"}
HEADERS_LASER = {"Authorization": f"Bearer {LASER_EXPERT_KEY}", "Content-Type": "application/json"}
HEADERS_ENDO = {"Authorization": f"Bearer {ENDO_EXPERT_KEY}", "Content-Type": "application/json"}
HEADERS_MED = {"Authorization": f"Bearer {MED_EXPERT_KEY}", "Content-Type": "application/json"}
HEADERS_INTEG = {"Authorization": f"Bearer {INTEG_KEY}", "Content-Type": "application/json"}


# ---------------------- Outpatient physician: clinical record ----------------------
class ReportGenerator:
    """Generates a structured clinical record from a fundus image (Methods 4.1.2)."""

    def __init__(self, json_path=FFA_CAPTIONS_JSON):
        self.model = REPORT_MODEL
        self.ffa_data = self._load_ffa_json(json_path)

    @staticmethod
    def _load_ffa_json(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load captions JSON: {e}", flush=True)
            return {}

    @staticmethod
    def _get_ffa_caption(image_name, ffa_data):
        base_name = os.path.splitext(image_name)[0]
        for key, caption in ffa_data.items():
            if base_name in key:
                return caption
        return "无相关辅助造影特征"

    @staticmethod
    def _generate_patient_vars(image_name, severity):
        """Draw demographics and metabolic history that the image cannot supply.

        Seeded on the image name so a given case is reproducible. Severe cases
        are drawn from longer-duration, higher-HbA1c distributions to reflect
        real care-seeking patterns.
        """
        seed = int(hashlib.md5(image_name.encode()).hexdigest(), 16) % 10**8
        random.seed(seed)

        is_severe = ("DR-3" in severity) or ("DR-4" in severity) \
            or ("重度" in severity) or ("增殖" in severity and "非增殖" not in severity)

        gender = random.choices(["男", "女"], weights=[0.55, 0.45])[0]
        age = random.randint(50, 80) if is_severe else int(random.gauss(60, 10))
        age = max(35, min(85, age))

        if is_severe:
            if random.random() < 0.85:
                duration = random.randint(15, 25)
                hba1c = round(random.uniform(9.0, 13.0), 1)
            else:
                duration = random.randint(3, 10)
                hba1c = round(random.uniform(6.5, 8.0), 1)
        else:
            duration = random.randint(3, 12)
            hba1c = round(random.uniform(6.5, 8.5), 1)

        if random.random() < 0.8:
            comorbidity = random.choice([
                "高血压", "糖尿病肾病", "高脂血症",
                "高血压合并肾病", "糖尿病周围神经病变", "否认其他常规合并症",
            ])
        else:
            comorbidity = "RANDOM_OPEN"

        visit_date = f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        return gender, age, duration, hba1c, comorbidity, visit_date

    @staticmethod
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_report(self, image_path: str, disease_type: str) -> str:
        image_name = Path(image_path).name
        patient_name = Path(image_path).stem

        ffa_caption = self._get_ffa_caption(image_name, self.ffa_data)
        gender, age, duration, hba1c, comorbidity, visit_date = \
            self._generate_patient_vars(image_name, disease_type)

        prompt_template = """
[输入参数]
- 疾病分级：{severity}
- 年龄：{age}岁
- 糖尿病病程：{duration}年
- 糖化血红蛋白(HbA1c)：{hba1c}%
- 合并症：{comorbidity} （注：若为"RANDOM_OPEN"，请结合临床随机生成一种合并症。且禁止出现"RANDOM_OPEN"等指令字眼）
- 【辅助特征（FFA造影英文描述）】：{ffa_caption}

[生成要求]
1. 跨模态病理转译：
   将提供的【辅助特征】转化为对应的"普通眼底彩照(CFP)表现"，并用客观中文写入病历！
   - 严禁使用造影专属词汇（如：荧光、渗漏、灌注、高荧光、遮蔽等）。
   - 必须结合你亲眼看到的彩照进行交叉验证，图文相符才可写入。
2. 事实约束：【眼部检查】部分必须严格符合眼底照图片。
3. 真实医学文书规范（极度重要）：输出的必须是一份严肃、可直接归档的电子门诊病历，以主治医生的第一人称客观视角陈述体征。**禁止**使用括号进行画外音解释。
4. 多样性发挥：结合上述参数，随机生成不重复的【主诉】和【现病史】，保持简明扼要。

请严格遵循下列格式输出中文眼底病例报告（不要输出大括号及提示语）：
# 眼底病病例报告

你的职责是根据提供给你的这些信息，生成符合该眼底图片的病例报告。请主要依据眼底图像生成。\
该眼底图片仅为单眼，图片名称中的left、right代表该图像为左眼或者右眼，仅对患病眼睛进行描述。其他未提供部分请按照患者情况进行适当生成。\
不同的图片来自不同的患者，患者的姓名即图片名称，所以请务必注意病历报告生成的个性化！不要千篇一律。在报告中，请务必涵盖以下内容：糖尿病视网膜病变的分期（轻度、中度、重度或增殖性）、视力状况的详细变化、眼底检查的具体表现（包括出血、渗出和新生血管的数量和位置）、糖尿病控制情况（如HbA1c水平）、既往眼科治疗史（激光、抗VEGF注射等）、黄斑区的状态、全身合并症（高血压、心脏病等）。请确保报告内容丰富多样，反映每位患者的独特情况。\
请遵循下列眼底病例报告模板：\
# 眼底病病例报告\
                ## 1. 基本信息
                - **姓名**：{patient_name}
                - **性别**：{gender}
                - **年龄**：{age}岁
                - **就诊日期**：{visit_date}
                ## 2. 主诉\
                - **主诉**：__________（眼别、症状、时间、就诊原因）\
                ## 3. 现病史\
                - **现病史**：__________\
                ## 4. 既往史（若无则写否认）\
                - **眼科病史**：__________\
                - **既往眼科手术**：__________\
                - **其他眼部疾病**：__________（如有）\
                - **全身性疾病**：\
                - **糖尿病**：__________（如有）\
                - **高血压**：__________（如有）\
                - **肾脏疾病**：__________（如有）\
                - **其他疾病**：__________（如有）\
                - **药物控制**：__________（如有）\
                ## 5. 眼部检查（左眼/右眼）\
                - **视力检查**：__________（裸眼视力/矫正视力）\
                - **眼压测量**：__________ mmHg\
                - **眼前段**：__________\
                - **玻璃体**：__________\
                - **视盘**：__________\
                - **黄斑**：__________\
                - **视网膜**：__________（如出血、新生血管等）\
                ## 6. 诊断\
                - **初步诊断**：__________\
"""
        final_prompt = prompt_template.format(
            patient_name=patient_name, gender=gender, severity=disease_type,
            age=age, duration=duration, hba1c=hba1c, comorbidity=comorbidity,
            ffa_caption=ffa_caption, visit_date=visit_date,
        )

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": final_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encode_image(image_path)}"}},
                ],
            }],
            "max_tokens": 10000,
            "temperature": 0.4,
        }
        headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}

        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(API_CALL_DELAY)
                resp = requests.post(f"{LLM_BASE_URL}/chat/completions",
                                     headers=headers, json=payload, timeout=240)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"  Report request failed ({attempt + 1}/{MAX_RETRIES}): {e}", flush=True)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return ""


# ------------------------- Style transfer and CSV export -------------------------
def extract_eye_exam_sections(content: str) -> Tuple[str, str, str]:
    """Pull the optic disc / macula / retina findings out of the record."""
    start = re.search(r"5\.\s*眼部检查", content)
    if not start:
        return "未找到描述", "未找到描述", "未找到描述"
    section = content[start.end():]
    end = re.search(r"6\.\s*诊断", section)
    if end:
        section = section[:end.start()]

    patterns = {
        "视盘": r"视盘[：:]\s*(.*?)(?=\n\s*[黄斑视网膜]|$)",
        "黄斑": r"黄斑[：:]\s*(.*?)(?=\n\s*[视盘视网膜]|$)",
        "视网膜": r"视网膜[：:]\s*(.*?)(?=\n\s*[视盘黄斑]|$)",
    }
    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, section, re.DOTALL)
        out[key] = m.group(1).strip() if m else "未找到描述"
    return out["视盘"], out["黄斑"], out["视网膜"]


def _parse_workflow_sse(response):
    """Dify workflows/run: the result arrives in the workflow_finished event."""
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
            return event.get("data", {}).get("outputs", {})
    return {}


def call_style_transfer(sentence: str) -> str:
    """Rewrite one finding using the DR terminology library (Methods 4.1.2)."""
    headers = {"Authorization": f"Bearer {STYLE_TRANSFER_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": {"GPT_report": sentence}, "response_mode": "streaming", "user": "style_user"}
    for _ in range(MAX_RETRIES):
        try:
            resp = requests.post(DIFY_WORKFLOW_URL, json=payload, headers=headers,
                                 stream=True, timeout=120)
            if resp.status_code == 200:
                outputs = _parse_workflow_sse(resp)
                # fin_report: rewritten; GPT_report: passthrough when retrieval was empty
                return outputs.get("fin_report") or outputs.get("GPT_report") or sentence
        except Exception:
            time.sleep(2)
    return sentence


def apply_style_transfer(file_path: str) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        disc, mac, retina = extract_eye_exam_sections(content)
        new_disc = call_style_transfer(f"视盘：{disc}").replace("视盘：", "").strip()
        new_mac = call_style_transfer(f"黄斑：{mac}").replace("黄斑：", "").strip()
        new_retina = call_style_transfer(f"视网膜：{retina}").replace("视网膜：", "").strip()
        content = (content
                   .replace(f"视盘：{disc}", f"视盘：{new_disc}")
                   .replace(f"黄斑：{mac}", f"黄斑：{new_mac}")
                   .replace(f"视网膜：{retina}", f"视网膜：{new_retina}"))
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception:
        return False


def build_csv_for_dr(dr_path: str, dr_type: str, output_csv: str):
    rows = []
    for pid in os.listdir(dr_path):
        folder = os.path.join(dr_path, pid)
        rf = os.path.join(folder, "眼底病病例报告.md")
        if not os.path.isdir(folder) or not os.path.exists(rf):
            continue
        with open(rf, "r", encoding="utf-8") as f:
            report = f.read()
        m = re.search(r'\*\*就诊日期\*\*：\s*(.*)', report) or re.search(r'就诊日期：\s*(.*)', report)
        rows.append([pid, dr_type, report, m.group(1).strip() if m else "未知日期"])
    if rows:
        pd.DataFrame(rows, columns=["id", "type", "report", "date"]).to_csv(
            output_csv, index=False, encoding="utf-8-sig")


# ------------------------------- File helpers -------------------------------
def save_to_file(folder, filename, content):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        f.write(content)


def read_from_file(folder, filename):
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _is_stage_valid(content: str) -> bool:
    return bool(content) and "失败" not in content and len(content.strip()) >= 300


# ---------------------------- Dify agent invocation ----------------------------
def _parse_sse(response):
    """Read a Dify chat-messages SSE stream. Returns (answer, conversation_id)."""
    chunks, conv_id = [], None
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
        etype = event.get("event", "")
        if etype == "message":
            if event.get("answer"):
                chunks.append(event["answer"])
        elif etype == "message_end":
            conv_id = event.get("conversation_id") or conv_id
            break
        if not conv_id:
            conv_id = event.get("conversation_id")
    answer = "".join(chunks)
    return (answer or None), conv_id


def call_agent(headers, data, name, conversation_id=None):
    """Call a Dify-hosted agent. conversation_id carries multi-round memory."""
    time.sleep(API_CALL_DELAY)
    if conversation_id:
        data["conversation_id"] = conversation_id

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(DIFY_CHAT_URL, headers=headers, json=data,
                                 stream=True, timeout=120)
            if resp.status_code == 200:
                answer, new_id = _parse_sse(resp)
                if answer:
                    return answer, new_id
                print(f"{name}: empty response", flush=True)
                return None, None
            print(f"{name}: HTTP {resp.status_code}", flush=True)
            if 500 <= resp.status_code < 600 and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return None, None
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return None, None
        except requests.exceptions.RequestException as e:
            print(f"{name}: {e}", flush=True)
            return None, None
    return None, None


def ask_laser_surgeon(uid, report, plan, query, conv_id):
    data = {"inputs": {"report": report, "eye_treatment": plan}, "query": query,
            "response_mode": "streaming", "user": uid, "files": []}
    return call_agent(HEADERS_LASER, data, "laser surgeon", conv_id)


def ask_endocrinologist(uid, report, plan, query, conv_id):
    data = {"inputs": {"report": report, "eye_treatment": plan}, "query": query,
            "response_mode": "streaming", "user": uid, "files": []}
    return call_agent(HEADERS_ENDO, data, "endocrinologist", conv_id)


def ask_pharmacist(uid, report, plan, query, conv_id):
    data = {"inputs": {"report": report, "eye_treatment": plan}, "query": query,
            "response_mode": "streaming", "user": uid, "files": []}
    return call_agent(HEADERS_MED, data, "pharmacist", conv_id)


def ask_ophthalmologist(uid, query, plan, patient, discussion, conv_id):
    data = {
        "inputs": {
            "id": patient["id"], "type": patient["type"], "report": patient["report"],
            "date": str(patient["date"]), "discussion": discussion, "current_plan": plan,
        },
        "query": query + "\n\n注意：请确保在您的响应中保留所有必要的眼科及内分泌综合治疗方案！",
        "response_mode": "streaming", "user": uid, "files": [],
    }
    return call_agent(HEADERS_EYE, data, "ophthalmologist", conv_id)


def nurse_consolidate(base_plan, new_input, task_type, pid):
    """Nurse agent: merge a discussion outcome into the plan, keeping its structure."""
    data = {
        "inputs": {"fundation_treatment": base_plan, "combined_treatment": new_input,
                   "type": task_type},
        "query": "请输出你整合好的方案",
        "response_mode": "streaming", "user": f"nurse_{pid}_{task_type}", "files": [],
    }
    time.sleep(API_CALL_DELAY)
    for attempt in range(3):
        try:
            resp = requests.post(DIFY_CHAT_URL, headers=HEADERS_INTEG, json=data,
                                 stream=True, timeout=120)
            if resp.status_code == 200:
                answer, _ = _parse_sse(resp)
                if answer:
                    return answer
            if 500 <= resp.status_code < 600 and attempt < 2:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"nurse ({task_type}): {e}", flush=True)
    return None


# ----------------------------- Multi-round discussion -----------------------------
def expert_discussion(ask_expert, uid, patient, plan, initial_query, follow_up_query,
                      max_rounds=MAX_ROUNDS):
    """One ophthalmologist-specialist discussion (Methods 4.2.3).

    Ends when the specialist raises no objection, or after max_rounds.
    On abort, the ophthalmologist's plan is retained.
    """
    current_plan = plan
    history = []
    expert_conv = eye_conv = None
    aborted = False

    objection_markers = ["否", "不完全", "不赞同", "不同意", "不支持", "不认同", "反对", "有异议"]

    for rnd in range(1, max_rounds + 1):
        query = initial_query if rnd == 1 else follow_up_query.format(feedback=history[-1])
        feedback, new_id = ask_expert(uid, patient["report"], current_plan, query, expert_conv)
        if not feedback:
            aborted = True
            break
        if new_id:
            expert_conv = new_id
        history.append(feedback)

        if not any(w in feedback for w in objection_markers):
            return current_plan, True   # consensus reached

        eye_query = (
            "您是否完全赞同该智能体专家提供的修改建议？请明确说明。\n"
            "如果您完全赞同，请直接提供修改后的完整治疗方案。\n"
            "如果您不完全赞同，请说明理由并给出您的方案。\n\n"
            "特别重要：请确保在您的响应中完整保留所有眼科治疗方案细节！不要省略或简化任何眼科治疗相关内容！\n"
            f"当前治疗方案：\n{current_plan}\n\n专家反馈：\n{feedback}"
        )
        eye_response, new_eye_id = ask_ophthalmologist(
            "ophthalmologist_user", eye_query, current_plan, patient, feedback, eye_conv)
        if not eye_response:
            aborted = True
            break
        if new_eye_id:
            eye_conv = new_eye_id
        current_plan = eye_response

    return (plan, False) if aborted else (current_plan, True)


# --------------------------- Treatment-plan distillation ---------------------------
def distil_plan(content: str, out_path: str) -> bool:
    """Reduce the final plan to the six-field treatment-plan template."""
    prompt = f"""请根据提供给你的治疗方案内容({content})，按照指定的模板提取每个部分的关键信息，务必确保内容完整、简洁明了！
注意：
1. 如果传入的方案内容为空、报错、或提示无法评估，请直接输出单句话："缺乏有效病情信息，无法生成治疗方案"，绝不能生成全篇都是"无"的机械列表。
2. 如果内容有效，填写时必须删除模板中的括号及提示语（如"包含细节参数"）。
3. 如果某一项内容确实没有，请填写"暂无"，不要自行编造。
填写模板（必须去除括号内容）：
1. 疾病诊断：\n2. 治疗原则：\n3. 治疗方案：\n   · 药物治疗\n   · 手术治疗(包含细节参数)\n4. 随访计划：\n5. 患者教育：\n6. 注意事项："""

    payload = {
        "model": DISTILL_MODEL,
        "messages": [
            {"role": "system",
             "content": "你的责任是对输入文本进行重新整理，确保提取关键信息，务必完整、简洁明了！务必完全遵循模版并注意避错指令！"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {DISTILL_API_KEY}", "Content-Type": "application/json"}

    for _ in range(MAX_RETRIES):
        try:
            time.sleep(API_CALL_DELAY)
            resp = requests.post(DISTILL_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(resp.json()["choices"][0]["message"]["content"])
                return True
        except Exception:
            time.sleep(2)
    return False


# --------------------------- Consultation for one patient ---------------------------
def _run_consultation(patient, pid, patient_dir):
    """Six stages, each cached so a failed run resumes where it stopped."""
    def cached(filename):
        if OVERWRITE_ALL:
            return None
        content = read_from_file(patient_dir, filename)
        return content if _is_stage_valid(content) else None

    # 1. Ophthalmologist drafts the initial plan
    initial_plan = cached(f"{pid}_initial.md")
    if not initial_plan:
        initial_plan, _ = ask_ophthalmologist(
            f"patient_{pid}",
            "基于患者的情况，提供个性化治疗建议。必须包含详细的眼科治疗方案！治疗方案中："
            "禁止出现考虑、需评估、需专科医师制定（你就是专科医师），"
            "禁止出现如果...则...类条件句，所有治疗决策必须呈现为肯定句",
            "", patient, "NONE", None,
        )
        if not initial_plan:
            return False, "initial plan"
        save_to_file(patient_dir, f"{pid}_initial.md", initial_plan)

    # 2. Laser surgeon discussion
    surgical_plan = cached(f"{pid}_surgical.md")
    if not surgical_plan:
        surgical_plan, ok = expert_discussion(
            ask_laser_surgeon, f"laser_{pid}", patient, initial_plan,
            "请严格基于以下当前提供的治疗方案中的手术部分进行评估。如需要进行激光光凝手术，您是否完全赞同该手术方案？"
            "如果完全赞同，请直接说明理由并结束讨论。如果不完全赞同，请提供原因并给出您的替代方案，以及相关的指南依据。\n\n",
            "依据上一轮反馈，您是否完全赞同当前手术方案？如果完全赞同，请结束讨论。"
            "如果不赞同，请给出替代方案及指南依据。\n\n",
        )
        if not ok:
            return False, "laser surgeon discussion"
        save_to_file(patient_dir, f"{pid}_surgical.md", surgical_plan)

    # 3. Endocrinologist discussion
    endocrine_plan = cached(f"{pid}_endocrine.md")
    if not endocrine_plan:
        endocrine_plan, ok = expert_discussion(
            ask_endocrinologist, f"endo_{pid}", patient, initial_plan,
            "请仅评估治疗方案中关于内分泌管理的部分：您是否完全赞同？\n"
            "1. 若完全赞同，直接说明结束讨论。\n"
            "2. 若不完全赞同，请提供修改建议（血糖目标、药物调整）及指南依据。\n"
            "3. 请勿修改眼科治疗方案部分。\n\n",
            "依据反馈，您是否完全赞同眼科专家提供治疗方案中的内分泌部分？请说明理由。\n\n",
        )
        if not ok:
            return False, "endocrinologist discussion"
        save_to_file(patient_dir, f"{pid}_endocrine.md", endocrine_plan)

    # 4. Nurse consolidates the two discussions
    combined_plan = cached(f"{pid}_combined.md")
    if not combined_plan:
        combined_plan = nurse_consolidate(surgical_plan, endocrine_plan, "combine", pid)
        if not combined_plan:
            return False, "nurse consolidation"
        save_to_file(patient_dir, f"{pid}_combined.md", combined_plan)

    # 5. Pharmacist reviews medication safety
    medicine_result = cached(f"{pid}_medicine.md")
    if not medicine_result:
        medicine_result, ok = expert_discussion(
            ask_pharmacist, f"pharm_{pid}", patient, combined_plan,
            "作为药学专家，审查药物使用：是否有禁忌症、剂量是否适当、是否有药物相互作用。"
            "您是否完全赞同药物治疗部分？请勿修改非药物相关内容（如眼科手术）。\n\n",
            "依据反馈，您是否完全赞同该方案中的药物治疗部分？请勿修改非药物相关内容。\n\n",
        )
        if not ok:
            return False, "pharmacist discussion"
        save_to_file(patient_dir, f"{pid}_medicine.md", medicine_result)

    # 6. Nurse produces the final plan
    final_plan = cached(f"{pid}_final.md")
    if not final_plan:
        final_plan = nurse_consolidate(combined_plan, medicine_result, "final", pid)
        if not final_plan:
            return False, "final consolidation"
        save_to_file(patient_dir, f"{pid}_final.md", final_plan)

    return True, None


def process_one_patient(patient, dr_name, dr_path):
    pid = str(patient["id"])
    patient_dir = os.path.join(dr_path, pid)
    os.makedirs(patient_dir, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        ok, stage = _run_consultation(patient, pid, patient_dir)
        if not ok:
            print(f"{pid}: attempt {attempt} failed at [{stage}]", flush=True)
            time.sleep(2)
            continue

        final_plan = read_from_file(patient_dir, f"{pid}_final.md")
        template_path = os.path.join(patient_dir, "treatment_plan.md")
        template = read_from_file(patient_dir, "treatment_plan.md")
        need_distil = OVERWRITE_ALL or not template or len(template.strip()) < 50

        if need_distil:
            if not distil_plan(final_plan, template_path):
                time.sleep(2)
                continue
            template = read_from_file(patient_dir, "treatment_plan.md")

        if template and len(template.strip()) > 50 and "缺乏有效病情" not in template:
            print(f"{pid}: done", flush=True)
            return True, None
        time.sleep(2)

    return False, {"id": pid, "dr": dr_name, "step": "consultation"}


# ---------------------------------- Entry point ----------------------------------
def main():
    all_dr_types = {
        "DR-1": "轻度非增殖性糖尿病视网膜病变",
        "DR-2": "中度非增殖性糖尿病视网膜病变",
        "DR-3": "重度非增殖性糖尿病视网膜病变",
        "DR-4": "增殖性糖尿病视网膜病变",
    }
    dr_types = {k: v for k, v in all_dr_types.items() if k in SELECTED_DR_TYPES}
    failures = []

    if RUN_STEPS_1_TO_3:
        generator = ReportGenerator()

        def make_record(img, dr_type, tmp_dir):
            base = Path(img).stem
            report_file = os.path.join(tmp_dir, f"{base}.md")
            for attempt in range(MAX_RETRIES):
                try:
                    report = generator.generate_report(img, dr_type)
                    if report and report.strip():
                        os.makedirs(tmp_dir, exist_ok=True)
                        with open(report_file, "w", encoding="utf-8") as f:
                            f.write(report)
                        apply_style_transfer(report_file)
                        return None
                except Exception as e:
                    print(f"  {base}: {e}", flush=True)
                time.sleep(2)
            return {"id": base, "step": "clinical record"}

        for image_root in IMAGE_ROOTS:
            for dr_name, dr_type in dr_types.items():
                dr_path = os.path.join(image_root, dr_name)
                if not os.path.isdir(dr_path):
                    continue
                images = [os.path.join(dr_path, f) for f in os.listdir(dr_path)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if not images:
                    continue

                tmp_dir = os.path.join(dr_path, "report")
                os.makedirs(tmp_dir, exist_ok=True)

                pending = []
                for img in images:
                    done_path = os.path.join(dr_path, Path(img).stem, "眼底病病例报告.md")
                    if OVERWRITE_ALL or not os.path.exists(done_path) \
                            or os.path.getsize(done_path) < 50:
                        pending.append(img)
                if SAMPLE_LIMIT:
                    pending = pending[:SAMPLE_LIMIT]

                csv_path = os.path.join(image_root, f"{dr_name}.csv")
                if not pending:
                    build_csv_for_dr(dr_path, dr_type, csv_path)
                    continue

                print(f"{dr_name}: generating {len(pending)} clinical records", flush=True)
                with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                    futures = [pool.submit(make_record, img, dr_type, tmp_dir) for img in pending]
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res:
                            res["dr"] = dr_name
                            failures.append(res)

                for img in pending:
                    base = Path(img).stem
                    folder = os.path.join(dr_path, base)
                    os.makedirs(folder, exist_ok=True)
                    src = os.path.join(tmp_dir, f"{base}.md")
                    if os.path.exists(src):
                        shutil.move(src, os.path.join(folder, "眼底病病例报告.md"))
                    shutil.copy(img, os.path.join(folder, base + Path(img).suffix))

                build_csv_for_dr(dr_path, dr_type, csv_path)
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if RUN_STEPS_4_TO_5:
        for image_root in IMAGE_ROOTS:
            for dr_name in dr_types:
                csv_path = os.path.join(image_root, f"{dr_name}.csv")
                if not os.path.exists(csv_path):
                    continue
                df = pd.read_csv(csv_path)
                dr_path = os.path.join(image_root, dr_name)
                print(f"{dr_name}: {len(df)} cases", flush=True)

                with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                    futures = [
                        pool.submit(process_one_patient,
                                    {"id": str(r["id"]), "type": r["type"],
                                     "report": r["report"], "date": r["date"]},
                                    dr_name, dr_path)
                        for _, r in df.iterrows()
                    ]
                    for fut in as_completed(futures):
                        ok, record = fut.result()
                        if not ok and record:
                            failures.append(record)

    if failures:
        print("\nFailed cases:", flush=True)
        for f in failures:
            print(f"  {f['id']} | {f['dr']} | {f['step']}", flush=True)
    else:
        print("\nAll cases processed.", flush=True)


if __name__ == "__main__":
    main()
