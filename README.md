# DRAgent

Code and data for **"An agentic multidisciplinary framework for diabetic
retinopathy treatment planning"**.

DRAgent is a role-based multi-agent system in which an ophthalmologist, a laser
surgeon, an endocrinologist and a pharmacist deliberate to produce an
individualized, guideline-grounded treatment plan from a fundus image and a
clinical record.

## Structure

```
agents/
  dsl/            Dify workflow for each clinical agent
  pipeline.py     Full pipeline: fundus image -> clinical record -> treatment plan
evaluation/
  dsl/                 Dify workflow for each guideline-compliance scorer
  loader.py            Locates plans and references for either output layout
  ncr_scoring.py       Guideline-compliance scoring (Supplementary Tables 1-2)
  distill.py           Structured distillation of plans before similarity
  objective_metrics.py Weighted BLEU / ROUGE-L / METEOR / BERTScore (Suppl. Table 3)
DR_treatment_planning_dataset/
  hybrid/              Hybrid real-synthetic cases (physician-revised)
  real/                De-identified real clinical cases (sample)
  large_data/          DRAgent-generated cases
```

## Agents

| Agent | Implementation |
|---|---|
| Outpatient physician | `ReportGenerator` in `agents/pipeline.py` |
| Ophthalmologist | `agents/dsl/ophthalmologist.yml` |
| Laser surgeon | `agents/dsl/laser_surgeon.yml` |
| Endocrinologist | `agents/dsl/endocrinologist.yml` |
| Pharmacist | `agents/dsl/pharmacist.yml` |
| Nurse | `agents/dsl/nurse.yml` |

`agents/dsl/style_transfer.yml` is not an agent; it applies the DR terminology
library during clinical-record construction.

Each specialist DSL follows the same structure: summarise the input, retrieve
from the role-specific guideline knowledge base, refine the retrieved text, then
reason within the role. Import each `.yml` into Dify and set the resulting app
key as an environment variable below.

## Setup

```bash
pip install -r requirements.txt

export LLM_API_KEY=...              # backbone for the outpatient physician
export DISTILL_API_KEY=...          # plan distillation
export DIFY_OPHTHALMOLOGIST_KEY=app-...
export DIFY_LASER_SURGEON_KEY=app-...
export DIFY_ENDOCRINOLOGIST_KEY=app-...
export DIFY_PHARMACIST_KEY=app-...
export DIFY_NURSE_KEY=app-...
export DIFY_STYLE_TRANSFER_KEY=app-...
export DR_DATA_ROOT=/path/to/data
```

Set `REPORT_MODEL` to switch the backbone (`gpt-5`, `gemini-2.5-flash`,
`deepseek-v3.1`, `qwen3.5-flash`); change the model in each DSL to match.

## Run

```bash
python agents/pipeline.py
```

Toggle `RUN_STEPS_1_TO_3` (image → clinical record) and `RUN_STEPS_4_TO_5`
(consultation → final plan) at the top of the file. Each stage is cached per
case, so an interrupted run resumes where it stopped.

## Evaluate

The unified evaluator has two explicit layouts:

- `--layout agent`: `--run-root` is one `agents/pipeline.py` output root and
  contains `DR-1` through `DR-4` directly. Reports normally sit beside plans.
  For a consolidated plan-only copy, pass the grade-stratified report tree with
  `--report-root` explicitly.
- `--layout llm`: `--run-root` contains `{model}/{DR-N}/{case_id}.md`, the flat
  output produced by the standalone-LLM generators (and used by HuatuoGPT-3).
  `--report-root` is required.

Both layouts use the same distillation, objective metrics, NCR criteria, and
aggregation code after case discovery.

Guideline compliance:

```bash
export DIFY_SCORER_OPHTHALMIC_KEY=app-...
export DIFY_SCORER_ENDOCRINE_KEY=app-...

python evaluation/ncr_scoring.py --scheme ophthalmic --layout agent \
    --run-root run/gpt5/ --model-name gpt5 --out ncr_ophth_agent.csv
python evaluation/ncr_scoring.py --scheme endocrine --layout llm \
    --run-root run/ --report-root reports/ --out ncr_endo_llm.csv
```

Objective similarity (distil first, then score):

```bash
export DISTILL_API_KEY=...

python evaluation/distill.py --layout agent --run-root run/gpt5/ --model-name gpt5
python evaluation/objective_metrics.py --layout agent --run-root run/gpt5/ \
    --model-name gpt5 --ref-root reference/ --out metrics_agent.csv
```

For an agent run, each clinical report is read from the pipeline case directory:
`{run-root}/{DR-N}/{case_id}/眼底病病例报告.md`. Physician-reference
abstracts use `reference/{DR-N}/{case_id}/治疗方案_abstract.md`.

Standalone-LLM plans use `{run-root}/{model}/{DR-N}/{case_id}.md`; their
distilled form is `{case_id}_abstract.md` in the same directory. Their reports
are read from `{report-root}/{DR-N}/{case_id}/眼底病病例报告.md`.

The scoring criteria are wrapped in two Dify workflows
(`evaluation/dsl/scorer_ophthalmic.yml`, `evaluation/dsl/scorer_endocrine.yml`),
scored by Claude Opus 4.8 — a third-party model independent of every system under
test. Each workflow first selects the criteria a case triggers, then returns one
TRUE/FALSE verdict per triggered criterion; `ncr_scoring.py` turns those verdicts
into the Normalized Compliance Rate. The criteria and the aggregation weights are
listed in Supplementary Tables 1-3.

## Data

`DR_treatment_planning_dataset/` contains three subsets, each a directory of
cases holding a clinical record and the corresponding treatment plan:

| Subset | Role in the study |
|---|---|
| `hybrid/` | Physician-revised cases; the primary evaluation benchmark |
| `real/` | Multi-center real clinical cases; independent validation |
| `large_data/` | DRAgent-generated cases; a training and fine-tuning resource |

Each case directory is named by its identifier in the source fundus-image
dataset (EyeQ/EyePACS, OIA-DDR, MMRDR, IDRiD, APTOS 2019); use this name to
retrieve the corresponding image from the original repository. The fundus images
themselves are not redistributed.

The complete dataset will be released upon acceptance.

## Knowledge base

The retrieval knowledge bases are built from the five published Chinese clinical
guidelines cited in the paper. The guideline texts are copyrighted and are not
included here; the `dataset_ids` in the DSLs refer to our own Dify workspace and
must be replaced with your own.

## Citation

A citation will be added once the paper is published.

## License

Code: MIT. Data: CC BY-NC 4.0.
