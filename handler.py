import os
import glob
import json
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

أعد JSON صالحًا فقط:
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
MATCHER_MAX_NEW_TOKENS = int(os.environ.get("MATCHER_MAX_NEW_TOKENS", "180"))
MATCHER_MIN_RELEVANT_SCORE = float(os.environ.get("MATCHER_MIN_RELEVANT_SCORE", "0.35"))

_MATCHER_MODEL = None
_MATCHER_TOKENIZER = None
_MATCHER_REGISTRY = None
_MATCHER_DEVICE = None


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
        parsed = extract_json_object(
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

    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable is missing."
        )

    job_input = job.get("input", {})

    if not isinstance(job_input, dict):
        return {
            "error": "RunPod input must be a JSON object."
        }

    request_type = job_input.get("type")

    # Runtime request route.
    if request_type == "advisory_match":

        if job_input.get("preflight", False):
            return advisory_match_preflight(
                job_input,
                token,
            )

        return advisory_match_inference(
            job_input,
            token,
        )

    # Training route.
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
        return training_preflight(
            training_type,
            checkpoint_path,
        )

    validate_training_files(
        training_type
    )

    # Avoid stale output if the same worker processes another run.
    if training_type == "matcher":
        shutil.rmtree(
            info["output_dir"],
            ignore_errors=True
        )

    os.makedirs(
        info["output_dir"],
        exist_ok=True
    )

    command = [
        "axolotl",
        "train",
        info["config"],
    ]

    # base/meta/specialist continue the same training run.
    # matcher v2 refines the current matcher LoRA weights via lora_model_dir.
    if info["resume"]:
        command.extend(
            [
                "--resume-from-checkpoint",
                checkpoint_path,
            ]
        )

    requested_epochs = job_input.get("num_epochs")

    if requested_epochs is not None:

        requested_epochs = int(requested_epochs)

        if requested_epochs < 1:
            raise ValueError(
                "num_epochs must be >= 1"
            )

        command.extend(
            [
                "--num-epochs",
                str(requested_epochs),
            ]
        )

    print(
        f"Starting {training_type} training...",
        flush=True
    )

    print(
        "Command: " + " ".join(command),
        flush=True
    )

    training_tail = run_command(
        command,
        stream=True,
    )

    new_checkpoint = latest_checkpoint(
        info["output_dir"]
    )

    print(
        f"New checkpoint: {new_checkpoint}",
        flush=True
    )

    commit_sha = push_checkpoint(
        training_type,
        repo_dir,
        new_checkpoint,
        git_env,
    )

    return {
        "status": "completed",
        "training_type": training_type,
        "checkpoint": os.path.basename(
            new_checkpoint
        ),
        "saved_to": info["target_checkpoint_rel"],
        "github_commit": commit_sha,
        "log_tail": training_tail[-4000:],
    }


runpod.serverless.start({
    "handler": handler
})
