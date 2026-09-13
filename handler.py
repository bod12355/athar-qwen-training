import os
import glob
import json
import re
import shutil
import subprocess
import time
from collections import deque

import runpod


ROOT = "/workspace/data/athar"

GITHUB_REPO = os.environ.get(
    "GITHUB_REPO",
    "da02bod-art/athar-qwen-training"
)

GITHUB_BRANCH = os.environ.get(
    "GITHUB_BRANCH",
    "main"
)

GIT_USER_NAME = os.environ.get(
    "GIT_USER_NAME",
    "da02bod-art"
)

GIT_USER_EMAIL = os.environ.get(
    "GIT_USER_EMAIL",
    "da02.bod@gmail.com"
)

MATCHER_REGISTRY_REL = "advisors/advisors_registry_v2.json"
MATCHER_PROMPT_REL = "prompts/matcher_system_prompt_v2.md"

# This is the exact per-candidate instruction used to build train_matcher_v1.jsonl.
MATCHER_CANDIDATE_SYSTEM_PROMPT = """أنت Athar OS Advisor Candidate Matcher.
مهمتك تقييم مدى ملاءمة مستشار واحد فقط لهذه المنظمة في وضعها الحالي، بالاعتماد على بيانات المنظمة والبرامج وملف المستشار.

لا تشترط أن تكتب المنظمة كلمة "تحتاج" أو تصف مشكلة صراحة. اعتبر ثلاثة أنواع من الأدلة:
1) حاجة أو فجوة أو مخاطرة مذكورة صراحة.
2) حاجة يمكن استنتاجها مباشرة من وقائع المدخل دون اختراع معلومات جديدة.
3) واقع تشغيلي أو برامجي يجعل خبرة المستشار ذات قيمة مادية واضحة الآن.

لا تعتبر مجرد إنجاز سابق سببًا كافيًا وحده لترشيح المستشار إذا لم توجد حاجة حالية أو فرصة أو مخاطرة أو تعقيد ذو صلة.
activation_conditions دليل إيجابي قوي، وscope_boundaries حد ملزم.
not_primary_when لا يعني الاستبعاد التلقائي، لكنه يمنع تضخيم الملاءمة إذا كان الدور الحقيقي يخص مستشارًا آخر.
ميّز بدقة بين المجالات المتجاورة، خصوصًا:
- KPI ولوحات القيادة مقابل MEAL وقياس الأثر.
- تصميم المبادرات مقابل التخطيط التشغيلي مقابل إدارة المحافظ والمشاريع.
- الحوكمة والامتثال مقابل القيادة التنفيذية.
- التشخيص المؤسسي مقابل التخطيط الاستراتيجي.
- التحول والتغيير مقابل الجودة.

أعد JSON صالحًا فقط، واجعل reason جملة واحدة لا تتجاوز 18 كلمة ولا تسرد أسماء المجالات:
{"relevant": true/false, "score": 0.0, "reason": "سبب عربي قصير ومحدد مستند إلى واقعة من المدخل"}

التقدير:
0.85-1.00 ملاءمة مباشرة ومحورية.
0.70-0.84 ملاءمة قوية وواضحة.
0.50-0.69 دور مساند مادي.
0.35-0.49 دور محدود فقط إذا كان له سبب محدد.
أقل من 0.35 يكون relevant=false عادة.
"""

MATCHER_BASE_MODEL = "Qwen/Qwen3-14B"
MATCHER_BATCH_SIZE = int(os.environ.get("MATCHER_BATCH_SIZE", "2"))
MATCHER_MAX_INPUT_TOKENS = int(os.environ.get("MATCHER_MAX_INPUT_TOKENS", "8192"))
MATCHER_MAX_NEW_TOKENS = int(os.environ.get("MATCHER_MAX_NEW_TOKENS", "96"))
MATCHER_MIN_RELEVANT_SCORE = float(os.environ.get("MATCHER_MIN_RELEVANT_SCORE", "0.35"))

_MATCHER_MODEL = None
_MATCHER_TOKENIZER = None
_MATCHER_REGISTRY = None
_MATCHER_DEVICE = None


# ---------------------------------------------------------------------
# Grounded routing engine v5
# Production advisory_match no longer depends on the binary Matcher LoRA.
# It uses the base Qwen3-14B as a two-stage grounded router:
#   1) extract advisory needs/opportunities with evidence IDs
#   2) compare ALL advisors together against those grounded needs
# ---------------------------------------------------------------------

ROUTER_MAX_INPUT_TOKENS = int(
    os.environ.get("ROUTER_MAX_INPUT_TOKENS", "20000")
)
ROUTER_NEEDS_MAX_NEW_TOKENS = int(
    os.environ.get("ROUTER_NEEDS_MAX_NEW_TOKENS", "900")
)
ROUTER_RANK_MAX_NEW_TOKENS = int(
    os.environ.get("ROUTER_RANK_MAX_NEW_TOKENS", "1400")
)
ROUTER_MIN_SCORE = float(
    os.environ.get("ROUTER_MIN_SCORE", "0.35")
)

_ROUTER_MODEL = None
_ROUTER_TOKENIZER = None
_ROUTER_REGISTRY = None
_ROUTER_DEVICE = None

NEEDS_SYSTEM_PROMPT = """أنت محلل احتياجات استشارية لمنظومة Athar OS.

ستستلم قائمة FACTS مرقمة مأخوذة من بيانات منظمة وبرامجها.
استخرج فقط الاحتياجات أو المخاطر أو الفرص الاستشارية التي يمكن دعمها مباشرة بهذه الوقائع.

قواعد ملزمة:
- لا تشترط وجود كلمة "تحتاج" أو "مشكلة".
- لا تخترع فجوة غير مدعومة.
- الإنجاز السابق وحده لا يعني وجود مشكلة حالية.
- يمكن اعتبار تعقيد حقيقي في المحفظة أو البرامج فرصة استشارية إذا كانت له قيمة مادية واضحة.
- كل حاجة يجب أن تشير إلى evidence_ids صحيحة من FACTS.
- اجمع الوقائع المتشابهة في حاجة واحدة بدل التكرار.
- أخرج من 0 إلى 8 احتياجات فقط.
- لا ترشح مستشارين في هذه المرحلة.

kind يجب أن يكون واحدًا من:
explicit_gap
direct_inference
advisory_opportunity

priority يجب أن يكون:
high
medium
low

أعد JSON فقط بهذا الشكل:
{
  "needs": [
    {
      "need_id": "N1",
      "need": "وصف عربي موجز للحاجة",
      "kind": "direct_inference",
      "priority": "high",
      "evidence_ids": ["F2", "P3"]
    }
  ]
}
"""

ROUTING_SYSTEM_PROMPT = """أنت محرك توجيه المستشارين في Athar OS.

ستستلم:
1) GROUNDED_NEEDS: احتياجات أو فرص أو تعقيدات موثقة، وكل واحدة مرتبطة بأدلة.
2) ADVISORS: ملفات التوجيه للمستشارين.

قيّم جميع المستشارين معًا، وليس كل مستشار بمعزل عن الآخرين.

المطلوب:
- رشح كل مستشار له قيمة مادية حقيقية لإحدى الاحتياجات الحالية.
- لا يوجد عدد ثابت للترشيحات.
- لا تضف مستشارًا لمجرد أن مجاله مهم عمومًا.
- لا تخترع احتياجًا جديدًا غير موجود في GROUNDED_NEEDS.
- كل مستشار مرشح يجب أن يحتوي matched_need_ids غير فارغة.
- activation_conditions دليل إيجابي قوي.
- scope_boundaries حد ملزم.
- not_primary_when لا يعني الاستبعاد التلقائي؛ قد يكون الدور supporting إذا أضاف قيمة مادية.
- قارن المجالات المتجاورة حتى لا تكرر نفس الحاجة بلا داعٍ.

تمييزات مهمة:
- Advisor 14: KPI وتعريف المؤشرات والمصادر وخطوط الأساس والمستهدفات ولوحات القيادة.
- Advisor 15: MEAL والنتائج والتقييم والتعلم ونظرية التغيير وقوة دليل الأثر والسببية.
- Advisor 11: تصميم التدخل أو المبادرة قبل اعتمادها واختبار الفرضيات والقيمة.
- Advisor 12: تحويل أعمال معتمدة إلى خطة تشغيلية وملاك وجدول وموارد واعتماديات.
- Advisor 13: المحافظ والبرامج والمشاريع وPMO والبوابات والمنافع وتعارض الموارد.
- Advisor 16: الحوكمة والامتثال والصلاحيات والسياسات والضوابط وأدلة التطبيق.
- Advisor 1: القيادة التنفيذية وحسم القرار وترتيب الأولويات والتنفيذ.
- Advisor 5: التحول والتبني والمقاومة وموجات التغيير؛ مجرد وجود ERP لا يكفي.
- Advisor 6: الجودة وعدم المطابقة والمعايير والإجراءات التصحيحية والتحسين المستمر.

الدرجات:
0.85-1.00 = مباشر ومحوري
0.70-0.84 = قوي وواضح
0.50-0.69 = مساند مادي
0.35-0.49 = محدود لكن مبرر
أقل من 0.35 لا ترشحه عادة.

role يجب أن يكون primary أو supporting.

أعد JSON فقط:
{
  "ranked": [
    {
      "advisor_id": 14,
      "score": 0.91,
      "role": "primary",
      "matched_need_ids": ["N2"],
      "reason": "سبب عربي قصير ومحدد"
    }
  ]
}
"""



# ---------------------------------------------------------------------
# Rich AI Router v8
# The model evaluates ALL 16 advisors together using rich Expert DNA.
# There is NO deterministic need gate and NO hard-coded advisor mapping.
# ---------------------------------------------------------------------

RICH_REGISTRY_PATH = os.environ.get(
    "RICH_REGISTRY_PATH",
    f"{ROOT}/data/advisors_registry_rich_v3.json"
)

RICH_MAX_INPUT_TOKENS = int(
    os.environ.get("RICH_MAX_INPUT_TOKENS", "20000")
)

RICH_MAX_NEW_TOKENS = int(
    os.environ.get("RICH_MAX_NEW_TOKENS", "1400")
)

RICH_MIN_RELEVANT_SCORE = float(
    os.environ.get("RICH_MIN_RELEVANT_SCORE", "0.40")
)

_RICH_MODEL = None
_RICH_TOKENIZER = None
_RICH_REGISTRY = None
_RICH_DEVICE = None

RICH_ROUTER_SYSTEM_PROMPT = """أنت Athar OS Rich Advisor Router.

هذه مرحلة اكتشاف الملاءمة قبل أن تختار الجمعية ستة مستشارين.
مهمتك تقييم الـ16 مستشارًا جميعًا باستخدام ملفات Expert DNA الغنية، ثم إرجاع كل المستشارين المناسبين فعلاً فقط.

قواعد حاسمة:
1) فكّر في الـ16 جميعًا قبل الإخراج.
2) لا يوجد عدد ثابت؛ قد يكون المناسب 2 أو 5 أو 8 أو أكثر.
3) لا تشترط وجود كلمة "تحتاج" أو "مشكلة". الملاءمة قد تأتي من:
   - فجوة أو مخاطرة صريحة.
   - حاجة مستنتجة مباشرة من الوقائع.
   - تعقيد تشغيلي/برامجي حقيقي.
   - فرصة تحسين مادية واضحة من واقع عمل المنظمة.
4) لا تعتبر الإنجاز السابق وحده دليلاً على وجود فجوة حالية.
5) Supporting مقبول فقط إذا كانت له قيمة مادية مستقلة، وليس ارتباطًا هامشيًا.
6) لا تطبق Minimum Expert Principle هنا؛ الاختيار النهائي لستة مستشارين يتم لاحقًا.
7) كل مستشار مختار يجب أن يستند إلى evidence_ids صحيحة من FACTS.
8) activation_when دليل إيجابي، وnot_primary_when وboundaries تمنع تضخيم الدور.
9) فرّق بدقة بين:
   - 14 KPI/Dashboard و15 MEAL/Impact.
   - 11 Initiative Design و12 Operational Planning و13 Portfolio/Program/Project.
   - 1 Executive Leadership و16 Governance/Compliance.
   - 2 Institutional Diagnosis و8 Strategic Planning.
   - 5 Change/Adoption و6 Quality/Continuous Improvement.
10) لا تُخرج المستشارين غير المناسبين.

معايرة score:
0.85-1.00 = ملاءمة محورية وواضحة
0.70-0.84 = ملاءمة قوية
0.50-0.69 = دور مساند مادي
0.40-0.49 = قيمة محدودة لكن حقيقية
أقل من 0.40 = لا تخرجه

role:
core = يعالج بعدًا رئيسيًا ظاهرًا في الحالة
supporting = يضيف بعدًا مساندًا ماديًا

أعد JSON صالحًا فقط، بدون Markdown، وبدون شرح خارج JSON.
استخدم المفتاح advisor_id حرفيًا كما هو.
اجعل reason جملة واحدة قصيرة جدًا، بحد أقصى 14 كلمة.
اجعل evidence_ids من 1 إلى 3 فقط.

الشكل المطلوب:
{
  "matches": [
    {
      "advisor_id": 13,
      "score": 0.91,
      "role": "core",
      "evidence_ids": ["F6", "P3"],
      "reason": "سبب عربي قصير ومحدد"
    }
  ]
}
"""


def build_rich_facts(organization, programs):

    facts = []

    def add_fact(fid, source, value):
        if value is None:
            return

        if isinstance(value, (list, dict)):
            text = json.dumps(
                value,
                ensure_ascii=False
            )
        else:
            text = str(value).strip()

        if text:
            facts.append({
                "fact_id": fid,
                "source": source,
                "text": text,
            })

    add_fact("F1", "organization.name", organization.get("name"))
    add_fact("F2", "organization.type", organization.get("type"))
    add_fact("F3", "organization.sector", organization.get("sector"))
    add_fact("F4", "organization.activity_fields", organization.get("activity_fields"))
    add_fact("F5", "organization.short_description", organization.get("short_description"))
    add_fact("F6", "organization.detailed_description", organization.get("detailed_description"))
    add_fact("F7", "organization.competitive_advantage", organization.get("competitive_advantage"))
    add_fact("F8", "organization.important_notes", organization.get("important_notes"))

    for index, program in enumerate(programs, start=1):
        if not isinstance(program, dict):
            continue

        parts = []

        for key in [
            "name",
            "type",
            "description",
            "target_audience",
            "beneficiary_value",
            "delivery_method",
        ]:
            value = program.get(key)

            if value is not None and str(value).strip():
                parts.append(
                    f"{key}={str(value).strip()}"
                )

        if parts:
            facts.append({
                "fact_id": f"P{index}",
                "source": f"programs[{index - 1}]",
                "text": " | ".join(parts),
            })

    return facts


def load_rich_registry():

    if not os.path.isfile(RICH_REGISTRY_PATH):
        raise RuntimeError(
            f"Rich advisor registry not found: {RICH_REGISTRY_PATH}"
        )

    with open(
        RICH_REGISTRY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        registry = json.load(file)

    advisors = registry.get("advisors", [])

    if len(advisors) != 16:
        raise RuntimeError(
            f"Rich registry must contain exactly 16 advisors; got {len(advisors)}"
        )

    return registry


def ensure_rich_router_model():

    global _RICH_MODEL
    global _RICH_TOKENIZER
    global _RICH_REGISTRY
    global _RICH_DEVICE

    if (
        _RICH_MODEL is not None
        and _RICH_TOKENIZER is not None
        and _RICH_REGISTRY is not None
    ):
        return

    print(
        "Loading Qwen3-14B for rich AI advisor routing...",
        flush=True,
    )

    started = time.time()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    _RICH_REGISTRY = load_rich_registry()

    tokenizer = AutoTokenizer.from_pretrained(
        MATCHER_BASE_MODEL,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MATCHER_BASE_MODEL,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )

    model.eval()

    _RICH_MODEL = model
    _RICH_TOKENIZER = tokenizer
    _RICH_DEVICE = next(model.parameters()).device

    print(
        f"Rich AI router ready in {round(time.time() - started, 2)}s",
        flush=True,
    )


def _extract_first_json_object(text):

    cleaned = (
        str(text)
        .strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start >= 0 and end > start:
        try:
            return json.loads(
                cleaned[start:end + 1]
            )
        except Exception:
            pass

    raise ValueError(
        f"Rich router did not return valid JSON. Raw: {cleaned[:1200]}"
    )


def generate_rich_evaluations(
    organization,
    programs,
):

    import torch

    facts = build_rich_facts(
        organization,
        programs,
    )

    payload = {
        "organization_name": organization.get("name"),
        "facts": facts,
        "advisors": _RICH_REGISTRY["advisors"],
    }

    messages = [
        {
            "role": "system",
            "content": RICH_ROUTER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
            ),
        },
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(
        encoded["attention_mask"].sum().item()
    )

    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Rich router input too long: "
            f"{input_tokens} > {RICH_MAX_INPUT_TOKENS}"
        )

    encoded = {
        key: value.to(_RICH_DEVICE)
        for key, value in encoded.items()
    }

    print(
        f"Rich routing input tokens: {input_tokens}",
        flush=True,
    )

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            no_repeat_ngram_size=8,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[
        :,
        encoded["input_ids"].shape[1]:,
    ]

    raw_text = _RICH_TOKENIZER.decode(
        generated[0],
        skip_special_tokens=True,
    )

    parsed = _extract_first_json_object(
        raw_text
    )

    return {
        "parsed": parsed,
        "facts": facts,
        "input_tokens": input_tokens,
        "raw_text": raw_text,
    }



def normalize_rich_matches(parsed, facts):

    valid_ids = set(range(1, 17))
    fact_ids = {fact["fact_id"] for fact in facts}

    raw_matches = parsed.get("matches", [])

    if not isinstance(raw_matches, list):
        raise ValueError("Rich router output missing matches list.")

    normalized = []
    seen = set()

    for row in raw_matches:

        if not isinstance(row, dict):
            continue

        advisor_value = None
        for key in ("advisor_id", " advisor_id", "adviser_id", " adviser_id"):
            if key in row:
                advisor_value = row.get(key)
                break

        try:
            advisor_id = int(advisor_value)
        except (TypeError, ValueError):
            continue

        if advisor_id not in valid_ids or advisor_id in seen:
            continue

        try:
            score = float(row.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        score = max(0.0, min(1.0, score))

        if score < RICH_MIN_RELEVANT_SCORE:
            continue

        role = str(row.get("role", "supporting")).strip().lower()

        if role not in {"core", "supporting"}:
            role = "supporting"

        evidence_ids = row.get("evidence_ids", [])

        if not isinstance(evidence_ids, list):
            evidence_ids = []

        evidence_ids = [
            str(fid)
            for fid in evidence_ids
            if str(fid) in fact_ids
        ][:3]

        if not evidence_ids:
            continue

        reason = re.sub(
            r"\s+",
            " ",
            str(row.get("reason", "")).strip(),
        )

        if not reason:
            reason = "ملاءمة مادية مدعومة بوقائع من بيانات المنظمة."

        if len(reason.split()) > 18:
            reason = " ".join(reason.split()[:18]).rstrip("،,.") + "."

        normalized.append({
            "advisor_id": advisor_id,
            "score": round(score, 4),
            "role": role,
            "evidence_ids": evidence_ids,
            "reason": reason,
        })

        seen.add(advisor_id)

    normalized.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return normalized


def advisory_match_rich_v8(job_input):

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_rich_router_model()

    print(
        "Evaluating all 16 advisors with rich Expert DNA...",
        flush=True,
    )

    result = generate_rich_evaluations(
        organization,
        programs,
    )

    ranked = normalize_rich_matches(
        result["parsed"],
        result["facts"],
    )

    response = {
        "status": "completed",
        "type": "advisory_match",
        "routing_engine": "rich_ai_v8_1",
        "model": MATCHER_BASE_MODEL,
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "evaluated_advisors": 16,
        "matched_advisors": len(ranked),
        "input_tokens": result["input_tokens"],
        "ranked": ranked,
    }

    if job_input.get("debug", False):
        response["raw_output"] = result["raw_text"]

    return response



RICH_V9_GROUPS = [
    [1, 2, 3, 4],
    [5, 6, 7, 16],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
]

RICH_V9_MAX_NEW_TOKENS = int(os.environ.get("RICH_V9_MAX_NEW_TOKENS", "420"))

RICH_V9_SYSTEM_PROMPT = """أنت Athar OS Rich Advisor Router v9.

ستستلم FACTS موثقة عن منظمة وبرامجها، وأربعة مستشارين بملفات Expert DNA غنية.
قيّم الأربعة جميعًا، ثم أخرج فقط المستشارين الذين لديهم قيمة مادية حقيقية الآن.

قواعد:
- لا يوجد عدد ثابت.
- لا تشترط كلمة "تحتاج" أو "مشكلة".
- يجوز الترشيح بسبب فجوة صريحة، حاجة مستنتجة مباشرة، تعقيد تشغيلي/برامجي حقيقي، أو فرصة تحسين مادية واضحة.
- لا تخترع مشكلة غير موجودة.
- الإنجاز السابق وحده لا يعني وجود حاجة حالية.
- وجود ERP أو إعادة هيكلة أو درجة حوكمة مرتفعة لا يعني تلقائيًا الحاجة لمستشار تغيير أو حوكمة.
- وجود برنامج قائم لا يعني تلقائيًا الحاجة لإعادة تصميمه.
- كثرة البرامج وتنوعها قد تدعم 12 أو 13 إذا كان التعقيد واضحًا.
- 15 يحتاج دليل نتائج/تقييم/أثر/تعلم، وليس مجرد وجود برامج.
- 14 يحتاج KPI/مصادر بيانات/خط أساس/مستهدفات/لوحات.
- 8 يحتاج قرارًا أو مراجعة أو مفاضلة استراتيجية حقيقية.
- 2 يحتاج تشخيص/نضج/جاهزية أو فجوة قدرة فعلية.
- 16 يحتاج فجوة/مخاطرة/قرار حوكمي أو امتثال فعلي.
- 5 يحتاج تحول/تبنٍ/مقاومة/انتقال فعلي.
- Supporting مقبول فقط إذا كانت له قيمة مادية مستقلة.
- كل ترشيح يجب أن يستند إلى 1-3 evidence_ids صحيحة.
- السبب يجب أن يوضح لماذا المستشار مناسب الآن.

الدرجات أعداد صحيحة:
85-100 محوري
70-84 قوي
50-69 مساند مادي
40-49 محدود لكنه حقيقي
أقل من 40 لا تخرجه

أخرج سطرًا واحدًا لكل مستشار مناسب فقط:
ADVISOR_ID|SCORE|ROLE|EVIDENCE_IDS|REASON

ROLE = core أو supporting
EVIDENCE_IDS مثال F6,P3
REASON جملة عربية قصيرة.

مثال:
13|92|core|F6,P3|تعدد البرامج وتنوعها يخلق حاجة فعلية لإدارة المحفظة والأولويات.

إذا لم يكن أي منهم مناسبًا اكتب:
NONE

ممنوع JSON وممنوع Markdown وممنوع أي شرح إضافي.
"""


def _rich_v9_advisor_map():
    return {int(a["advisor_id"]): a for a in _RICH_REGISTRY["advisors"]}


def _parse_rich_v9_lines(text, allowed_advisor_ids, valid_fact_ids):
    cleaned = str(text).replace("```text", "").replace("```", "").strip()

    if not cleaned or cleaned.upper() == "NONE":
        return []

    results = []
    seen = set()

    for raw_line in cleaned.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line.strip())
        if not line or line.upper() == "NONE":
            continue

        parts = line.split("|", 4)
        if len(parts) != 5:
            continue

        a_raw, s_raw, role_raw, ev_raw, reason_raw = [x.strip() for x in parts]
        a_m = re.search(r"\d+", a_raw)
        s_m = re.search(r"\d+(?:\.\d+)?", s_raw)

        if not a_m or not s_m:
            continue

        advisor_id = int(a_m.group())
        if advisor_id not in allowed_advisor_ids or advisor_id in seen:
            continue

        score = float(s_m.group())
        if score <= 1:
            score *= 100
        score = max(0.0, min(100.0, score))
        if score < 40:
            continue

        role = role_raw.lower()
        if role not in {"core", "supporting"}:
            role = "supporting"

        evidence_ids = []
        for token in re.split(r"[,،;\s]+", ev_raw):
            token = token.strip().upper()
            if token in valid_fact_ids and token not in evidence_ids:
                evidence_ids.append(token)
        evidence_ids = evidence_ids[:3]

        if not evidence_ids:
            continue

        reason = re.sub(r"\s+", " ", reason_raw).strip()
        if not reason:
            continue
        if len(reason.split()) > 18:
            reason = " ".join(reason.split()[:18]).rstrip("،,.") + "."

        results.append({
            "advisor_id": advisor_id,
            "score": round(score / 100.0, 4),
            "role": role,
            "evidence_ids": evidence_ids,
            "reason": reason,
        })
        seen.add(advisor_id)

    return results


def generate_rich_v9_group(facts, advisors):
    import torch

    messages = [
        {"role": "system", "content": RICH_V9_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(
            {"facts": facts, "advisors": advisors},
            ensure_ascii=False
        )},
    ]

    prompt = _RICH_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _RICH_TOKENIZER(prompt, return_tensors="pt", add_special_tokens=False)
    input_tokens = int(encoded["attention_mask"].sum().item())

    if input_tokens > RICH_MAX_INPUT_TOKENS:
        raise ValueError(f"Rich v9 group input too long: {input_tokens}")

    encoded = {k: v.to(_RICH_DEVICE) for k, v in encoded.items()}

    with torch.inference_mode():
        output_ids = _RICH_MODEL.generate(
            **encoded,
            max_new_tokens=RICH_V9_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=10,
            eos_token_id=_RICH_TOKENIZER.eos_token_id,
            pad_token_id=_RICH_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated = output_ids[:, encoded["input_ids"].shape[1]:]
    text = _RICH_TOKENIZER.decode(generated[0], skip_special_tokens=True)
    return text, input_tokens


def advisory_match_rich_v9(job_input):
    organization, programs = normalize_advisory_input(job_input.get("input", {}))
    ensure_rich_router_model()

    facts = build_rich_facts(organization, programs)
    valid_fact_ids = {f["fact_id"] for f in facts}
    advisor_map = _rich_v9_advisor_map()

    all_matches = []
    debug_groups = []
    total_tokens = 0

    print("Rich v9: evaluating all 16 advisors in four groups...", flush=True)

    for idx, group_ids in enumerate(RICH_V9_GROUPS, start=1):
        print(f"Rich v9 group {idx}/4: {group_ids}", flush=True)

        raw_text, input_tokens = generate_rich_v9_group(
            facts,
            [advisor_map[i] for i in group_ids],
        )
        total_tokens += input_tokens

        parsed = _parse_rich_v9_lines(
            raw_text,
            set(group_ids),
            valid_fact_ids,
        )
        all_matches.extend(parsed)

        if job_input.get("debug", False):
            debug_groups.append({
                "group": group_ids,
                "input_tokens": input_tokens,
                "raw_output": raw_text,
                "parsed_matches": parsed,
            })

    all_matches.sort(key=lambda x: x["score"], reverse=True)

    response = {
        "status": "completed",
        "type": "advisory_match",
        "routing_engine": "rich_ai_v9",
        "model": MATCHER_BASE_MODEL,
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "evaluated_advisors": 16,
        "matched_advisors": len(all_matches),
        "total_input_tokens": total_tokens,
        "ranked": all_matches,
    }

    if job_input.get("debug", False):
        response["group_debug"] = debug_groups

    return response


RUNS = {
    "base": {
        "config": f"{ROOT}/configs/base_config.yaml",
        "source_checkpoint_rel": "checkpoints/base",
        "target_checkpoint_rel": "checkpoints/base",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-qlora",
        "resume": True,
    },

    "meta": {
        "config": f"{ROOT}/configs/meta_config.yaml",
        "source_checkpoint_rel": "checkpoints/meta",
        "target_checkpoint_rel": "checkpoints/meta",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-meta-qlora",
        "resume": True,
    },

    "specialist": {
        "config": f"{ROOT}/configs/specialist_config.yaml",
        "source_checkpoint_rel": "checkpoints/specialist",
        "target_checkpoint_rel": "checkpoints/specialist",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-specialist-qlora",
        "resume": True,
    },

    # New task:
    # Start from Meta adapter weights, but train as a NEW task/run.
    "matcher": {
        "config": f"{ROOT}/configs/matcher_config.yaml",
        "source_checkpoint_rel": "checkpoints/matcher",
        "target_checkpoint_rel": "checkpoints/matcher",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-matcher-v2-qlora",
        "resume": False,
    },
}


def run_command(cmd, cwd=None, env=None, stream=False):

    if stream:

        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        tail = deque(maxlen=200)

        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line)

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                "Command failed:\n" +
                "".join(tail)[-8000:]
            )

        return "".join(tail)

    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stdout[-8000:]
        )

    return result.stdout


def clone_repo_without_lfs(token, repo_dir="/tmp/athar_training_repo"):

    shutil.rmtree(
        repo_dir,
        ignore_errors=True
    )

    clone_url = (
        f"https://x-access-token:{token}"
        f"@github.com/{GITHUB_REPO}.git"
    )

    env = os.environ.copy()

    # Clone normal files but skip all heavy LFS files initially.
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"

    run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            GITHUB_BRANCH,
            clone_url,
            repo_dir,
        ],
        env=env,
    )

    return repo_dir, env


def clone_source_checkpoint(training_type, token):

    info = RUNS[training_type]

    repo_dir, env = clone_repo_without_lfs(token)

    run_command(
        ["git", "lfs", "install", "--local"],
        cwd=repo_dir,
        env=env,
    )

    source_checkpoint_rel = info["source_checkpoint_rel"]

    # Download only the source adapter needed for this run.
    run_command(
        [
            "git",
            "lfs",
            "pull",
            f"--include={source_checkpoint_rel}/**",
            "--exclude=",
        ],
        cwd=repo_dir,
        env=env,
    )

    checkpoint_path = os.path.join(
        repo_dir,
        source_checkpoint_rel
    )

    adapter_file = os.path.join(
        checkpoint_path,
        "adapter_model.safetensors"
    )

    if not os.path.exists(adapter_file):
        raise RuntimeError(
            "adapter_model.safetensors was not downloaded."
        )

    # Detect an LFS pointer accidentally being used as model weights.
    if os.path.getsize(adapter_file) < 10_000_000:
        raise RuntimeError(
            "Checkpoint appears to be a Git LFS pointer, "
            "not the real adapter file."
        )

    return repo_dir, checkpoint_path, env


def load_matcher_assets(repo_dir):

    registry_path = os.path.join(
        repo_dir,
        MATCHER_REGISTRY_REL
    )

    prompt_path = os.path.join(
        repo_dir,
        MATCHER_PROMPT_REL
    )

    if not os.path.exists(registry_path):
        raise RuntimeError(
            f"Matcher registry not found: {MATCHER_REGISTRY_REL}"
        )

    if not os.path.exists(prompt_path):
        raise RuntimeError(
            f"Matcher prompt not found: {MATCHER_PROMPT_REL}"
        )

    with open(
        registry_path,
        "r",
        encoding="utf-8"
    ) as f:
        registry = json.load(f)

    with open(
        prompt_path,
        "r",
        encoding="utf-8"
    ) as f:
        matcher_prompt = f.read()

    advisors = registry.get("advisors")

    if not isinstance(advisors, list):
        raise RuntimeError(
            "Matcher registry field 'advisors' must be a list."
        )

    if len(advisors) != 16:
        raise RuntimeError(
            f"Expected 16 advisors, found {len(advisors)}."
        )

    advisor_ids = [
        advisor.get("advisor_id")
        for advisor in advisors
    ]

    if advisor_ids != list(range(1, 17)):
        raise RuntimeError(
            f"Advisor IDs must be 1..16. Found: {advisor_ids}"
        )

    return registry, matcher_prompt



def clone_matcher_checkpoint(token):

    repo_dir, env = clone_repo_without_lfs(
        token,
        repo_dir="/tmp/athar_inference_repo",
    )

    run_command(
        ["git", "lfs", "install", "--local"],
        cwd=repo_dir,
        env=env,
    )

    checkpoint_rel = RUNS["matcher"]["target_checkpoint_rel"]

    run_command(
        [
            "git",
            "lfs",
            "pull",
            f"--include={checkpoint_rel}/**",
            "--exclude=",
        ],
        cwd=repo_dir,
        env=env,
    )

    checkpoint_path = os.path.join(
        repo_dir,
        checkpoint_rel,
    )

    adapter_file = os.path.join(
        checkpoint_path,
        "adapter_model.safetensors",
    )

    adapter_config = os.path.join(
        checkpoint_path,
        "adapter_config.json",
    )

    if not os.path.exists(adapter_file):
        raise RuntimeError(
            "Matcher adapter_model.safetensors was not downloaded."
        )

    if os.path.getsize(adapter_file) < 10_000_000:
        raise RuntimeError(
            "Matcher checkpoint appears to be a Git LFS pointer."
        )

    if not os.path.exists(adapter_config):
        raise RuntimeError(
            "Matcher adapter_config.json was not found."
        )

    return repo_dir, checkpoint_path


def compact_candidate(advisor):

    return {
        "advisor_id": advisor.get("advisor_id"),
        "name_ar": advisor.get("name_ar"),
        "mission_summary": advisor.get("mission_summary", ""),
        "core_scope": advisor.get("core_scope", []),
        "activation_conditions": advisor.get("activation_conditions", []),
        "not_primary_when": advisor.get("not_primary_when", []),
        "scope_boundaries": advisor.get("scope_boundaries", ""),
        "match_signals": advisor.get("match_signals", []),
    }


def ensure_matcher_model(token):

    global _MATCHER_MODEL
    global _MATCHER_TOKENIZER
    global _MATCHER_REGISTRY
    global _MATCHER_DEVICE

    if (
        _MATCHER_MODEL is not None
        and _MATCHER_TOKENIZER is not None
        and _MATCHER_REGISTRY is not None
    ):
        return

    print(
        "Loading matcher model and adapter...",
        flush=True,
    )

    started = time.time()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import PeftModel

    repo_dir, checkpoint_path = clone_matcher_checkpoint(
        token
    )

    registry, _ = load_matcher_assets(
        repo_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MATCHER_BASE_MODEL,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MATCHER_BASE_MODEL,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )

    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=False,
    )

    model.eval()

    _MATCHER_MODEL = model
    _MATCHER_TOKENIZER = tokenizer
    _MATCHER_REGISTRY = registry
    _MATCHER_DEVICE = next(model.parameters()).device

    print(
        f"Matcher ready in {round(time.time() - started, 2)}s",
        flush=True,
    )


def build_candidate_prompt(
    organization,
    programs,
    advisor,
):

    user_payload = {
        "organization": organization,
        "programs": programs,
        "candidate_advisor": compact_candidate(advisor),
    }

    messages = [
        {
            "role": "system",
            "content": MATCHER_CANDIDATE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                user_payload,
                ensure_ascii=False,
            ),
        },
    ]

    return _MATCHER_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_json_object(text):

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "", 1)
        text = text.replace("```", "", 1).strip()

    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(
                text[index:]
            )

            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"Model did not return valid JSON. Raw output: {text[:500]}"
    )



def parse_candidate_output(text):

    # First try strict JSON parsing.
    try:
        return extract_json_object(text)
    except ValueError:
        pass

    # Robust fallback for truncated/repetitive generations:
    # recover the classification and score even if "reason" was not closed.
    relevant_match = re.search(
        r'"relevant"\s*:\s*(true|false)',
        text,
        flags=re.IGNORECASE,
    )

    score_match = re.search(
        r'"score"\s*:\s*(-?\d+(?:\.\d+)?)',
        text,
    )

    reason_match = re.search(
        r'"reason"\s*:\s*"([^"]*)',
        text,
        flags=re.DOTALL,
    )

    if relevant_match is None or score_match is None:
        raise ValueError(
            f"Could not recover matcher classification. Raw output: {text[:500]}"
        )

    relevant = (
        relevant_match.group(1).lower() == "true"
    )

    score = float(
        score_match.group(1)
    )

    reason = ""

    if reason_match is not None:
        reason = re.sub(
            r"\s+",
            " ",
            reason_match.group(1),
        ).strip()

        # Keep a runaway unfinished reason from polluting the API.
        words = reason.split()

        if len(words) > 24:
            reason = " ".join(words[:24]).rstrip("،,.") + "."

    if not reason:
        reason = (
            "تم استرجاع التصنيف والدرجة من استجابة غير مكتملة."
        )

    return {
        "relevant": relevant,
        "score": score,
        "reason": reason,
    }


def normalize_candidate_result(
    advisor_id,
    raw_result,
):

    relevant = raw_result.get("relevant", False)

    if isinstance(relevant, str):
        relevant = relevant.strip().lower() == "true"

    relevant = bool(relevant)

    try:
        score = float(
            raw_result.get("score", 0.0)
        )
    except (TypeError, ValueError):
        score = 0.0

    score = max(
        0.0,
        min(1.0, score),
    )

    reason = str(
        raw_result.get("reason", "")
    ).strip()

    return {
        "advisor_id": advisor_id,
        "relevant": relevant,
        "score": round(score, 4),
        "reason": reason,
    }


def evaluate_advisor_batch(
    prompts,
    advisors,
):

    import torch

    encoded = _MATCHER_TOKENIZER(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    prompt_token_counts = encoded[
        "attention_mask"
    ].sum(dim=1).tolist()

    too_long = [
        {
            "advisor_id": advisors[index]["advisor_id"],
            "tokens": int(token_count),
        }
        for index, token_count in enumerate(
            prompt_token_counts
        )
        if token_count > MATCHER_MAX_INPUT_TOKENS
    ]

    if too_long:
        raise ValueError(
            "Matcher input exceeds safe token limit: "
            + json.dumps(
                too_long,
                ensure_ascii=False,
            )
        )

    encoded = {
        key: value.to(_MATCHER_DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        output_ids = _MATCHER_MODEL.generate(
            **encoded,
            max_new_tokens=MATCHER_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=8,
            eos_token_id=_MATCHER_TOKENIZER.eos_token_id,
            pad_token_id=_MATCHER_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated_ids = output_ids[
        :,
        encoded["input_ids"].shape[1]:,
    ]

    texts = _MATCHER_TOKENIZER.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    results = []

    for advisor, text in zip(
        advisors,
        texts,
    ):
        parsed = parse_candidate_output(
            text
        )

        normalized = normalize_candidate_result(
            advisor["advisor_id"],
            parsed,
        )

        normalized["input_tokens"] = int(
            prompt_token_counts[len(results)]
        )

        results.append(
            normalized
        )

    return results


def advisory_match_inference(
    job_input,
    token,
):

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_matcher_model(
        token
    )

    advisors = _MATCHER_REGISTRY["advisors"]

    evaluations = []

    for start in range(
        0,
        len(advisors),
        MATCHER_BATCH_SIZE,
    ):

        batch_advisors = advisors[
            start:start + MATCHER_BATCH_SIZE
        ]

        prompts = [
            build_candidate_prompt(
                organization,
                programs,
                advisor,
            )
            for advisor in batch_advisors
        ]

        print(
            "Evaluating advisors: "
            + ", ".join(
                str(a["advisor_id"])
                for a in batch_advisors
            ),
            flush=True,
        )

        batch_results = evaluate_advisor_batch(
            prompts,
            batch_advisors,
        )

        evaluations.extend(
            batch_results
        )

    ranked = [
        {
            "advisor_id": row["advisor_id"],
            "score": row["score"],
            "reason": row["reason"],
        }
        for row in evaluations
        if (
            row["relevant"]
            and row["score"] >= MATCHER_MIN_RELEVANT_SCORE
        )
    ]

    ranked.sort(
        key=lambda row: row["score"],
        reverse=True,
    )

    response = {
        "status": "completed",
        "type": "advisory_match",
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "evaluated_advisors": len(evaluations),
        "ranked": ranked,
    }

    if job_input.get("debug", False):
        response["evaluations"] = evaluations

    return response



def ensure_grounded_router_model(token):

    global _ROUTER_MODEL
    global _ROUTER_TOKENIZER
    global _ROUTER_REGISTRY
    global _ROUTER_DEVICE

    if (
        _ROUTER_MODEL is not None
        and _ROUTER_TOKENIZER is not None
        and _ROUTER_REGISTRY is not None
    ):
        return

    print(
        "Loading grounded routing model...",
        flush=True,
    )

    started = time.time()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    # We need only text assets from GitHub for production routing.
    # No Matcher LoRA is loaded here.
    repo_dir, _ = clone_repo_without_lfs(
        token,
        repo_dir="/tmp/athar_grounded_router_repo",
    )

    registry, _ = load_matcher_assets(
        repo_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MATCHER_BASE_MODEL,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MATCHER_BASE_MODEL,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )

    model.eval()

    _ROUTER_MODEL = model
    _ROUTER_TOKENIZER = tokenizer
    _ROUTER_REGISTRY = registry
    _ROUTER_DEVICE = next(model.parameters()).device

    print(
        f"Grounded router ready in {round(time.time() - started, 2)}s",
        flush=True,
    )


def build_grounded_facts(
    organization,
    programs,
):

    facts = []

    org_fields = [
        ("F1", "short_description"),
        ("F2", "detailed_description"),
        ("F3", "competitive_advantage"),
        ("F4", "important_notes"),
    ]

    for fact_id, field in org_fields:
        value = organization.get(field)

        if value is None:
            continue

        if isinstance(value, (list, dict)):
            value = json.dumps(
                value,
                ensure_ascii=False,
            )
        else:
            value = str(value).strip()

        if value:
            facts.append({
                "fact_id": fact_id,
                "source": f"organization.{field}",
                "text": value,
            })

    activity_fields = organization.get(
        "activity_fields",
        []
    )

    if activity_fields:
        facts.append({
            "fact_id": "F5",
            "source": "organization.activity_fields",
            "text": json.dumps(
                activity_fields,
                ensure_ascii=False,
            ),
        })

    for index, program in enumerate(
        programs,
        start=1,
    ):

        if not isinstance(program, dict):
            continue

        parts = []

        for field in [
            "name",
            "type",
            "description",
            "target_audience",
            "beneficiary_value",
            "delivery_method",
        ]:
            value = program.get(field)

            if value is None:
                continue

            value = str(value).strip()

            if value:
                parts.append(
                    f"{field}={value}"
                )

        if parts:
            facts.append({
                "fact_id": f"P{index}",
                "source": f"programs[{index - 1}]",
                "text": " | ".join(parts),
            })

    return facts


def generate_json_with_router(
    system_prompt,
    payload,
    max_new_tokens,
):

    import torch

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
            ),
        },
    ]

    prompt = _ROUTER_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = _ROUTER_TOKENIZER(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_tokens = int(
        encoded["attention_mask"].sum().item()
    )

    if input_tokens > ROUTER_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Grounded router input is too long: "
            f"{input_tokens} tokens > {ROUTER_MAX_INPUT_TOKENS}"
        )

    encoded = {
        key: value.to(_ROUTER_DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        output_ids = _ROUTER_MODEL.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.06,
            no_repeat_ngram_size=10,
            eos_token_id=_ROUTER_TOKENIZER.eos_token_id,
            pad_token_id=_ROUTER_TOKENIZER.pad_token_id,
            use_cache=True,
        )

    generated_ids = output_ids[
        :,
        encoded["input_ids"].shape[1]:,
    ]

    text = _ROUTER_TOKENIZER.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    parsed = extract_json_object(
        text
    )

    return parsed, input_tokens, text


def normalize_grounded_needs(
    raw_needs,
    facts,
):

    fact_map = {
        fact["fact_id"]: fact
        for fact in facts
    }

    allowed_kinds = {
        "explicit_gap",
        "direct_inference",
        "advisory_opportunity",
    }

    allowed_priorities = {
        "high",
        "medium",
        "low",
    }

    normalized = []

    if not isinstance(raw_needs, list):
        return normalized

    for index, raw_need in enumerate(
        raw_needs[:8],
        start=1,
    ):

        if not isinstance(raw_need, dict):
            continue

        need_text = str(
            raw_need.get("need", "")
        ).strip()

        if not need_text:
            continue

        evidence_ids = raw_need.get(
            "evidence_ids",
            []
        )

        if not isinstance(
            evidence_ids,
            list,
        ):
            evidence_ids = []

        evidence_ids = [
            str(evidence_id)
            for evidence_id in evidence_ids
            if str(evidence_id) in fact_map
        ]

        # A need without a valid evidence pointer is rejected.
        if not evidence_ids:
            continue

        need_id = f"N{len(normalized) + 1}"

        kind = raw_need.get(
            "kind",
            "direct_inference",
        )

        if kind not in allowed_kinds:
            kind = "direct_inference"

        priority = raw_need.get(
            "priority",
            "medium",
        )

        if priority not in allowed_priorities:
            priority = "medium"

        normalized.append({
            "need_id": need_id,
            "need": need_text,
            "kind": kind,
            "priority": priority,
            "evidence_ids": evidence_ids,
            "evidence": [
                {
                    "fact_id": evidence_id,
                    "source": fact_map[evidence_id]["source"],
                    "text": fact_map[evidence_id]["text"],
                }
                for evidence_id in evidence_ids
            ],
        })

    return normalized



def _text_blob(organization, programs):
    parts = []
    for field in [
        "name", "type", "sector", "short_description",
        "detailed_description", "competitive_advantage", "important_notes",
    ]:
        value = organization.get(field)
        if value:
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            parts.append(str(value))

    activity_fields = organization.get("activity_fields", [])
    if activity_fields:
        parts.append(json.dumps(activity_fields, ensure_ascii=False))

    for program in programs:
        if not isinstance(program, dict):
            continue
        for field in [
            "name", "type", "description",
            "target_audience", "beneficiary_value", "delivery_method",
        ]:
            value = program.get(field)
            if value:
                parts.append(str(value))

    return " ".join(parts)


def _contains_any(text, terms):
    return any(term in text for term in terms)


def extract_grounded_advisory_needs(organization, programs):
    """
    v6: deterministic, evidence-first signal extraction.
    No LLM is allowed to decide whether the organization has "a need".
    This prevents annual-report language from collapsing to needs=[].
    """

    facts = build_grounded_facts(organization, programs)
    fact_map = {f["fact_id"]: f for f in facts}
    blob = _text_blob(organization, programs)
    blob_lower = blob.lower()
    signals = []

    def add_signal(signal_type, statement, evidence_ids, priority):
        valid = [eid for eid in evidence_ids if eid in fact_map]
        if not valid:
            return
        if any(x.get("signal_type") == signal_type for x in signals):
            return

        signals.append({
            "need_id": f"N{len(signals) + 1}",
            "need": statement,
            "kind": "advisory_opportunity",
            "priority": priority,
            "signal_type": signal_type,
            "evidence_ids": valid,
            "evidence": [
                {
                    "fact_id": eid,
                    "source": fact_map[eid]["source"],
                    "text": fact_map[eid]["text"],
                }
                for eid in valid
            ],
        })

    # 1) Portfolio/program complexity is directly observable.
    program_count = len(programs)
    if program_count >= 6:
        ids = [f"P{i}" for i in range(1, min(program_count, 6) + 1)]
        add_signal(
            "portfolio_complexity",
            f"وجود {program_count} برنامجًا/مبادرة متنوعة يخلق تعقيدًا ماديًا في إدارة المحفظة والبرامج والأولويات والمنافع والتنسيق بينها.",
            ids,
            "high" if program_count >= 10 else "medium",
        )

    # 2) Seasonal / time-bound operational complexity.
    seasonal_terms = [
        "موسم", "موسمية", "رمضان", "الحج", "حاج", "الحجاج",
        "ضيف الرحمن", "ضيوف الرحمن", "بداية العام الدراسي", "صيفي",
    ]
    seasonal_ids = []
    for i, program in enumerate(programs, start=1):
        ptext = " ".join(
            str(program.get(k, ""))
            for k in ["name", "description", "delivery_method"]
        )
        if _contains_any(ptext, seasonal_terms):
            seasonal_ids.append(f"P{i}")

    if len(seasonal_ids) >= 2:
        add_signal(
            "operational_coordination",
            "وجود عدة برامج موسمية أو مقيدة بتوقيتات تنفيذية مختلفة يخلق حاجة مادية للتخطيط التشغيلي والتنسيق بين الجداول والملاك والموارد.",
            seasonal_ids[:6],
            "medium",
        )

    # 3) MEAL / impact only when explicit language exists.
    meal_terms = [
        "قياس الأثر", "إدارة الأثر", "الأثر الاجتماعي",
        "تقييم الأثر", "نتائج البرامج", "نظرية التغيير",
        "متابعة وتقييم", "المتابعة والتقييم", "meal",
    ]
    if _contains_any(blob_lower, [x.lower() for x in meal_terms]):
        ids = [
            f["fact_id"] for f in facts
            if _contains_any(f["text"].lower(), [x.lower() for x in meal_terms])
        ][:6]
        add_signal(
            "impact_measurement",
            "توجد إشارة صريحة إلى قياس الأثر أو توجيه البرامج لدعمه، ما يبرر مراجعة إطار النتائج والتقييم والتعلم وقوة دليل الأثر.",
            ids,
            "high",
        )

    # 4) KPI only when explicit KPI/dashboard language exists.
    kpi_terms = [
        "مؤشرات الأداء", "مؤشر أداء", "kpi", "لوحة قيادة",
        "dashboard", "خط الأساس", "المستهدفات", "مصدر بيانات",
    ]
    if _contains_any(blob_lower, [x.lower() for x in kpi_terms]):
        ids = [
            f["fact_id"] for f in facts
            if _contains_any(f["text"].lower(), [x.lower() for x in kpi_terms])
        ][:6]
        add_signal(
            "kpi_management",
            "توجد إشارات صريحة إلى مؤشرات الأداء أو مصادرها أو خطوط الأساس أو لوحات القيادة، ما يبرر دعم منظومة KPI واتخاذ القرار.",
            ids,
            "high",
        )

    # 5) Governance: require an actual current gap/risk, not an achievement.
    governance_terms = ["حوكمة", "امتثال", "صلاحيات", "سياسات", "إجراءات"]
    gap_terms = [
        "ضعف", "غياب", "غير واضح", "تعارض", "قصور",
        "مخالفة", "عدم امتثال", "تحتاج", "بحاجة", "مطلوب",
    ]
    for fact in facts:
        text = fact["text"]
        if _contains_any(text, governance_terms) and _contains_any(text, gap_terms):
            add_signal(
                "governance_gap",
                "توجد فجوة أو مخاطرة حوكمة/امتثال مذكورة صراحة وتحتاج ضبط الصلاحيات أو السياسات أو أدلة التطبيق.",
                [fact["fact_id"]],
                "high",
            )
            break

    # 6) Change management: only on adoption/resistance evidence.
    change_terms = [
        "مقاومة التغيير", "ضعف التبني", "عدم التبني", "رفض النظام",
        "صعوبة التغيير", "إدارة التغيير", "تحديات التبني",
    ]
    if _contains_any(blob, change_terms):
        ids = [
            f["fact_id"] for f in facts
            if _contains_any(f["text"], change_terms)
        ][:6]
        add_signal(
            "change_adoption",
            "توجد إشارات صريحة إلى تحديات تبني أو مقاومة تغيير تتطلب إدارة تغيير منظمة.",
            ids,
            "high",
        )

    # 7) Strategy: require explicit strategic review/development language.
    strategy_terms = [
        "خطة استراتيجية", "استراتيجية", "أهداف استراتيجية",
        "أولويات استراتيجية", "قضايا استراتيجية",
    ]
    strategy_action_terms = [
        "تحديث", "مراجعة", "إعادة", "غير واضحة",
        "تحتاج", "بحاجة", "مطلوب", "تطوير",
    ]
    for fact in facts:
        text = fact["text"]
        if _contains_any(text, strategy_terms) and _contains_any(text, strategy_action_terms):
            add_signal(
                "strategy",
                "توجد حاجة أو فرصة استراتيجية صريحة تتعلق بمراجعة أو تطوير الاتجاه والأهداف والأولويات.",
                [fact["fact_id"]],
                "high",
            )
            break

    # 8) Stakeholders / partnerships when explicitly present.
    partnership_terms = [
        "شراكات", "شركاء", "أصحاب المصلحة",
        "الجهات المانحة", "مانحين",
    ]
    if _contains_any(blob, partnership_terms):
        ids = [
            f["fact_id"] for f in facts
            if _contains_any(f["text"], partnership_terms)
        ][:6]
        add_signal(
            "stakeholders_partnerships",
            "توجد شراكات أو أطراف مصلحة متعددة بما يجعل إدارة العلاقة والقيمة المتبادلة مجالًا استشاريًا ماديًا.",
            ids,
            "medium",
        )

    return {
        "needs": signals[:8],
        "facts": facts,
        "input_tokens": 0,
        "raw_text": None,
    }


def compact_routing_advisor(advisor):

    return {
        "advisor_id": advisor.get("advisor_id"),
        "name_ar": advisor.get("name_ar"),
        "mission_summary": advisor.get(
            "mission_summary",
            "",
        ),
        "owned_outcome": advisor.get(
            "owned_outcome",
            "",
        ),
        "core_scope": advisor.get(
            "core_scope",
            [],
        ),
        "activation_conditions": advisor.get(
            "activation_conditions",
            [],
        ),
        "not_primary_when": advisor.get(
            "not_primary_when",
            [],
        ),
        "scope_boundaries": advisor.get(
            "scope_boundaries",
            "",
        ),
        "match_signals": advisor.get(
            "match_signals",
            [],
        ),
    }


def route_all_advisors(
    needs,
):

    advisors = [
        compact_routing_advisor(
            advisor
        )
        for advisor in _ROUTER_REGISTRY["advisors"]
    ]

    routing_needs = [
        {
            "need_id": need["need_id"],
            "need": need["need"],
            "kind": need["kind"],
            "priority": need["priority"],
            "evidence": [
                evidence["text"]
                for evidence in need["evidence"]
            ],
        }
        for need in needs
    ]

    raw, input_tokens, raw_text = generate_json_with_router(
        ROUTING_SYSTEM_PROMPT,
        {
            "grounded_needs": routing_needs,
            "advisors": advisors,
        },
        ROUTER_RANK_MAX_NEW_TOKENS,
    )

    return {
        "raw_ranked": raw.get(
            "ranked",
            []
        ),
        "input_tokens": input_tokens,
        "raw_text": raw_text,
    }


def validate_global_ranking(
    raw_ranked,
    needs,
):

    valid_advisor_ids = {
        advisor["advisor_id"]
        for advisor in _ROUTER_REGISTRY["advisors"]
    }

    valid_need_ids = {
        need["need_id"]
        for need in needs
    }

    seen_advisors = set()
    ranked = []

    if not isinstance(
        raw_ranked,
        list,
    ):
        return ranked

    for row in raw_ranked:

        if not isinstance(row, dict):
            continue

        try:
            advisor_id = int(
                row.get("advisor_id")
            )
        except (TypeError, ValueError):
            continue

        if (
            advisor_id not in valid_advisor_ids
            or advisor_id in seen_advisors
        ):
            continue

        matched_need_ids = row.get(
            "matched_need_ids",
            []
        )

        if not isinstance(
            matched_need_ids,
            list,
        ):
            matched_need_ids = []

        matched_need_ids = [
            str(need_id)
            for need_id in matched_need_ids
            if str(need_id) in valid_need_ids
        ]

        # This is the key guardrail:
        # no advisor can be returned without a grounded need.
        if not matched_need_ids:
            continue

        try:
            score = float(
                row.get("score", 0.0)
            )
        except (TypeError, ValueError):
            continue

        score = max(
            0.0,
            min(1.0, score),
        )

        if score < ROUTER_MIN_SCORE:
            continue

        role = str(
            row.get("role", "supporting")
        ).strip().lower()

        if role not in {
            "primary",
            "supporting",
        }:
            role = "supporting"

        reason = str(
            row.get("reason", "")
        ).strip()

        if not reason:
            reason = (
                "ملاءمة مرتبطة بحاجة موثقة في بيانات المنظمة."
            )

        words = reason.split()

        if len(words) > 30:
            reason = (
                " ".join(words[:30]).rstrip(
                    "،,."
                )
                + "."
            )

        ranked.append({
            "advisor_id": advisor_id,
            "score": round(score, 4),
            "role": role,
            "matched_need_ids": matched_need_ids,
            "reason": reason,
        })

        seen_advisors.add(
            advisor_id
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked


def advisory_match_grounded_v5(
    job_input,
    token,
):

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    ensure_grounded_router_model(
        token
    )

    print(
        "Stage 1/2: building deterministic grounded advisory signals...",
        flush=True,
    )

    extraction = extract_grounded_advisory_needs(
        organization,
        programs,
    )

    needs = extraction["needs"]

    print(
        f"Grounded advisory signals built: {len(needs)}",
        flush=True,
    )

    if not needs:
        response = {
            "status": "completed",
            "type": "advisory_match",
            "routing_engine": "grounded_v6",
            "run_id": job_input.get("run_id"),
            "organization_name": organization.get("name"),
            "needs_count": 0,
            "ranked": [],
        }

        if job_input.get("debug", False):
            response["needs"] = []
            response["need_extraction_input_tokens"] = extraction[
                "input_tokens"
            ]

        return response

    print(
        "Stage 2/2: comparing all advisors together...",
        flush=True,
    )

    routing = route_all_advisors(
        needs
    )

    ranked = validate_global_ranking(
        routing["raw_ranked"],
        needs,
    )

    response = {
        "status": "completed",
        "type": "advisory_match",
        "routing_engine": "grounded_v6",
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "needs_count": len(needs),
        "ranked": [
            {
                "advisor_id": row["advisor_id"],
                "score": row["score"],
                "reason": row["reason"],
            }
            for row in ranked
        ],
    }

    if job_input.get("debug", False):
        response["needs"] = needs
        response["routing_details"] = ranked
        response["need_extraction_input_tokens"] = extraction[
            "input_tokens"
        ]
        response["routing_input_tokens"] = routing[
            "input_tokens"
        ]

    return response


def normalize_advisory_input(raw_input):

    # Prefer a real JSON object, but temporarily accept
    # a JSON string for backend compatibility.
    if isinstance(raw_input, str):
        try:
            raw_input = json.loads(raw_input)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"advisory_match input is not valid JSON: {exc}"
            )

    if not isinstance(raw_input, dict):
        raise ValueError(
            "advisory_match input must be a JSON object."
        )

    organization = raw_input.get("organization")
    programs = raw_input.get("programs", [])

    if not isinstance(organization, dict):
        raise ValueError(
            "input.organization must be a JSON object."
        )

    if not isinstance(programs, list):
        raise ValueError(
            "input.programs must be a JSON array."
        )

    return organization, programs


def advisory_match_preflight(job_input, token):

    repo_dir, _ = clone_repo_without_lfs(token)

    registry, matcher_prompt = load_matcher_assets(
        repo_dir
    )

    organization, programs = normalize_advisory_input(
        job_input.get("input", {})
    )

    advisors = registry["advisors"]

    return {
        "status": "matcher_preflight_ok",
        "type": "advisory_match",
        "run_id": job_input.get("run_id"),
        "organization_name": organization.get("name"),
        "programs_count": len(programs),
        "advisors_count": len(advisors),
        "advisor_ids": [
            advisor["advisor_id"]
            for advisor in advisors
        ],
        "registry_version": registry.get("version"),
        "matcher_prompt_chars": len(matcher_prompt),
        "note": (
            "Matcher assets and request schema are valid. "
            "Matcher assets and request schema are valid. "
            "The trained matcher adapter is available for inference."
        ),
    }


def validate_training_files(training_type):

    info = RUNS[training_type]

    if not os.path.exists(info["config"]):
        raise RuntimeError(
            f"Config not found: {info['config']}"
        )

    extra = {}

    if training_type == "matcher":

        train_path = f"{ROOT}/data/train_matcher_v2.jsonl"
        validation_path = f"{ROOT}/data/validation_matcher_v2.jsonl"

        if not os.path.exists(train_path):
            raise RuntimeError(
                f"Matcher train dataset not found: {train_path}"
            )

        if not os.path.exists(validation_path):
            raise RuntimeError(
                f"Matcher validation dataset not found: {validation_path}"
            )

        def count_jsonl(path):
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Invalid JSONL in {path} at line "
                            f"{line_number}: {exc}"
                        )

                    if not isinstance(row.get("messages"), list):
                        raise RuntimeError(
                            f"Missing messages list in {path} "
                            f"at line {line_number}"
                        )

                    count += 1

            return count

        extra = {
            "train_samples": count_jsonl(train_path),
            "validation_samples": count_jsonl(validation_path),
            "train_dataset": train_path,
            "validation_dataset": validation_path,
        }

    return extra


def latest_checkpoint(output_dir):

    paths = glob.glob(
        os.path.join(
            output_dir,
            "checkpoint-*"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No checkpoint found in {output_dir}"
        )

    return max(
        paths,
        key=lambda path: int(
            os.path.basename(path).split("-")[-1]
        )
    )


def push_checkpoint(
    training_type,
    repo_dir,
    new_checkpoint,
    env,
):

    target_checkpoint_rel = RUNS[training_type][
        "target_checkpoint_rel"
    ]

    destination = os.path.join(
        repo_dir,
        target_checkpoint_rel
    )

    shutil.rmtree(
        destination,
        ignore_errors=True
    )

    shutil.copytree(
        new_checkpoint,
        destination
    )

    run_command(
        [
            "git",
            "config",
            "user.name",
            GIT_USER_NAME,
        ],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        [
            "git",
            "config",
            "user.email",
            GIT_USER_EMAIL,
        ],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        [
            "git",
            "add",
            target_checkpoint_rel,
        ],
        cwd=repo_dir,
        env=env,
    )

    status = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--quiet",
        ],
        cwd=repo_dir,
        env=env,
    )

    if status.returncode == 0:
        return "No checkpoint changes detected"

    run_command(
        [
            "git",
            "commit",
            "-m",
            f"Update {training_type} checkpoint from RunPod Serverless",
        ],
        cwd=repo_dir,
        env=env,
    )

    run_command(
        [
            "git",
            "push",
            "origin",
            GITHUB_BRANCH,
        ],
        cwd=repo_dir,
        env=env,
        stream=True,
    )

    commit_sha = run_command(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=repo_dir,
        env=env,
    ).strip()

    return commit_sha


def training_preflight(
    training_type,
    checkpoint_path,
):

    info = RUNS[training_type]

    adapter_file = os.path.join(
        checkpoint_path,
        "adapter_model.safetensors"
    )

    extra = validate_training_files(
        training_type
    )

    response = {
        "status": "training_preflight_ok",
        "training_type": training_type,
        "config": info["config"],
        "resume_mode": info["resume"],
        "source_checkpoint": checkpoint_path,
        "target_checkpoint_rel": info["target_checkpoint_rel"],
        "adapter_size_mb": round(
            os.path.getsize(adapter_file) / 1024 / 1024,
            2
        ),
    }

    response.update(extra)

    if training_type == "matcher":
        response["note"] = (
            "Matcher v2 will refine the existing matcher adapter "
            "using lora_model_dir and will start a NEW "
            "optimizer/scheduler state. It will NOT use "
            "--resume-from-checkpoint."
        )

    return response


def handler(job):

    job_input = job.get("input", {})

    if not isinstance(job_input, dict):
        return {
            "error": "RunPod input must be a JSON object."
        }

    request_type = job_input.get("type")

    # Production inference no longer needs GitHub access.
    if request_type == "advisory_match":

        if job_input.get("preflight", False):
            registry = load_rich_registry()

            return {
                "status": "advisory_match_preflight_ok",
                "routing_engine": "rich_ai_v9",
                "model": MATCHER_BASE_MODEL,
                "registry_path": RICH_REGISTRY_PATH,
                "advisor_count": len(
                    registry.get("advisors", [])
                ),
            }

        return advisory_match_rich_v9(
            job_input
        )

    # Training routes still require GitHub.
    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable is missing."
        )

    training_type = job_input.get(
        "training_type"
    )

    if training_type not in RUNS:
        return {
            "error": (
                "Provide type='advisory_match' "
                "or training_type must be "
                "base, meta, specialist, or matcher."
            )
        }

    info = RUNS[training_type]

    print(
        f"Preparing {training_type} training...",
        flush=True
    )

    repo_dir, checkpoint_path, git_env = clone_source_checkpoint(
        training_type,
        token,
    )

    print(
        f"Source checkpoint: {checkpoint_path}",
        flush=True
    )

    if job_input.get("preflight", False):
        return training_preflight_response(
            training_type,
            checkpoint_path,
        )

    clean_previous_outputs(
        info["output_dir"]
    )

    cmd = [
        "/workspace/axolotl-venv/bin/accelerate",
        "launch",
        "-m",
        "axolotl.cli.train",
        info["config"],
    ]

    if info["resume"]:
        cmd.extend([
            "--resume-from-checkpoint",
            checkpoint_path,
        ])

    env = os.environ.copy()

    env["PYTHONUNBUFFERED"] = "1"

    print(
        f"Starting {training_type} training...",
        flush=True
    )

    log_tail = run_command(
        cmd,
        cwd=ROOT,
        env=env,
        stream=True,
    )

    latest_checkpoint = find_latest_checkpoint(
        info["output_dir"]
    )

    print(
        f"Latest checkpoint: {latest_checkpoint}",
        flush=True
    )

    save_checkpoint_to_repo(
        repo_dir,
        latest_checkpoint,
        info["target_checkpoint_rel"],
    )

    commit_sha = push_checkpoint_to_github(
        repo_dir,
        info["target_checkpoint_rel"],
        training_type,
        git_env,
    )

    return {
        "status": "completed",
        "training_type": training_type,
        "checkpoint": os.path.basename(
            latest_checkpoint
        ),
        "saved_to": info["target_checkpoint_rel"],
        "github_commit": commit_sha,
        "log_tail": log_tail,
    }



runpod.serverless.start({
    "handler": handler
})
