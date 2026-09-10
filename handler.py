import os
import glob
import json
import shutil
import subprocess
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


RUNS = {
    "base": {
        "config": f"{ROOT}/configs/base_config.yaml",
        "checkpoint_rel": "checkpoints/base",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-qlora",
    },

    "meta": {
        "config": f"{ROOT}/configs/meta_config.yaml",
        "checkpoint_rel": "checkpoints/meta",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-meta-qlora",
    },

    "specialist": {
        "config": f"{ROOT}/configs/specialist_config.yaml",
        "checkpoint_rel": "checkpoints/specialist",
        "output_dir": f"{ROOT}/outputs/qwen3-14b-athar-specialist-qlora",
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


def clone_repo_without_lfs(token):

    repo_dir = "/tmp/athar_training_repo"

    shutil.rmtree(
        repo_dir,
        ignore_errors=True
    )

    clone_url = (
        f"https://x-access-token:{token}"
        f"@github.com/{GITHUB_REPO}.git"
    )

    env = os.environ.copy()

    # Clone text/config files, but do not download all LFS checkpoints.
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


def clone_checkpoint(training_type, token):

    repo_dir, env = clone_repo_without_lfs(token)

    run_command(
        ["git", "lfs", "install", "--local"],
        cwd=repo_dir,
        env=env,
    )

    checkpoint_rel = RUNS[training_type]["checkpoint_rel"]

    # Download only the selected checkpoint.
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
        checkpoint_rel
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


def normalize_advisory_input(raw_input):

    # Prefer a real JSON object, but accept a JSON string
    # temporarily for backend compatibility.
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
            "Actual ranking is intentionally disabled until "
            "the matcher adapter is trained."
        ),
    }


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

    checkpoint_rel = RUNS[training_type]["checkpoint_rel"]

    destination = os.path.join(
        repo_dir,
        checkpoint_rel
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
            checkpoint_rel,
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

    # New Athar OS matching route.
    if request_type == "advisory_match":

        if not job_input.get("preflight", False):
            return {
                "status": "matcher_not_trained_yet",
                "type": "advisory_match",
                "run_id": job_input.get("run_id"),
                "error": (
                    "The advisory_match API route is wired, "
                    "but actual ranking is disabled until the "
                    "new matcher adapter is trained."
                ),
            }

        return advisory_match_preflight(
            job_input,
            token,
        )

    # Existing training route remains unchanged.
    training_type = job_input.get(
        "training_type"
    )

    if training_type not in RUNS:
        return {
            "error": (
                "Provide type='advisory_match' "
                "or training_type must be "
                "base, meta, or specialist."
            )
        }

    info = RUNS[training_type]

    print(
        f"Preparing {training_type} training...",
        flush=True
    )

    repo_dir, checkpoint_path, git_env = clone_checkpoint(
        training_type,
        token,
    )

    print(
        f"Checkpoint: {checkpoint_path}",
        flush=True
    )

    if job_input.get("preflight", False):
        adapter_file = os.path.join(
            checkpoint_path,
            "adapter_model.safetensors"
        )

        return {
            "status": "preflight_ok",
            "training_type": training_type,
            "checkpoint": checkpoint_path,
            "adapter_size_mb": round(
                os.path.getsize(adapter_file) / 1024 / 1024,
                2
            ),
        }

    os.makedirs(
        info["output_dir"],
        exist_ok=True
    )

    command = [
        "axolotl",
        "train",
        info["config"],
        "--resume-from-checkpoint",
        checkpoint_path,
    ]

    # Optional override for future continuation runs.
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
        "github_commit": commit_sha,
        "log_tail": training_tail[-4000:],
    }


runpod.serverless.start({
    "handler": handler
})
