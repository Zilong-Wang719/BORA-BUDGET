from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.common import dump_json, load_config, load_problem_split
from bora.eval import summarize_records
from bora.llm import _ensure_tokenizer_padding, _render_prompt_with_tokenizer, detect_prompt_template


MC_COT_PROMPT = (
    "Please reason carefully about the multiple-choice question. "
    "At the end, put only the final option letter within \\boxed{{}}.\n\n"
    "Question:\n{question}"
)

MC_BOXED_FIRST_PROMPT = (
    "Answer the multiple-choice question. The first line of your response must be "
    "exactly the final option letter inside \\boxed{{}}, e.g. \\boxed{{A}}. "
    "After that, give at most three short sentences of explanation if needed.\n\n"
    "Question:\n{question}"
)


def _build_user_prompt(
    question: str,
    enable_thinking: bool | None,
    prompt_style: str = "cot",
) -> tuple[str, str]:
    suffix = "/think" if enable_thinking is True else "/no_think"
    template = MC_BOXED_FIRST_PROMPT if prompt_style == "boxed_first" else MC_COT_PROMPT
    user_prompt = f"{template.format(question=question).rstrip()}\n\n{suffix}"
    return user_prompt, suffix


def _choice_labels(problem: dict) -> list[str]:
    metadata = dict(problem.get("metadata") or {})
    labels = metadata.get("choice_labels") or ["A", "B", "C", "D", "E"]
    return [str(label).strip().upper() for label in labels if str(label).strip()]


def _extract_mc_answer(text: str, labels: list[str]) -> str | None:
    if not text:
        return None
    label_alt = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text, flags=re.IGNORECASE)
    for raw in reversed(boxed_matches):
        match = re.search(rf"\b({label_alt})\b", raw.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    patterns = [
        rf"(?:final\s+answer|answer)\s*(?:is|:)?\s*(?:option|choice)?\s*\(?({label_alt})\)?\b",
        rf"(?:option|choice)\s*\(?({label_alt})\)?\b",
        rf"\btherefore[,:\s]+(?:the\s+)?(?:answer\s+is\s+)?\(?({label_alt})\)?\b",
    ]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-20:]):
        cleaned = re.sub(r"[*`_~#>\[\]]", " ", line)
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        if re.fullmatch(rf"\(?({label_alt})\)?[.。!！]?", cleaned.strip(), flags=re.IGNORECASE):
            return re.sub(r"[^A-Za-z0-9]", "", cleaned).upper()
    return None


def _chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partial-output", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--prompt-style",
        choices=["cot", "boxed_first"],
        default="cot",
        help="Prompt style. boxed_first is useful for short no-thinking MC audits.",
    )
    parser.add_argument("--store-completion-text", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if args.enable_thinking and args.disable_thinking:
        raise ValueError("Choose at most one of --enable-thinking and --disable-thinking.")

    loaded_config = load_config(args.config)
    random_seed = int(args.random_seed if args.random_seed is not None else loaded_config.get("random_seed", 0))
    config = {**loaded_config, "mode": "eval", "random_seed": random_seed}
    llm_cfg = dict(config.get("llm", {}))
    solver_cfg = {**llm_cfg, **config.get("solver", {})}
    enable_thinking = True if args.enable_thinking else False if args.disable_thinking else solver_cfg.get("enable_thinking")

    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else solver_cfg.get("standard_cot_max_new_tokens", 1024))
    max_model_len = int(args.max_model_len if args.max_model_len is not None else max(max_new_tokens + 1024, int(llm_cfg.get("max_context_tokens", 4096))))
    model_name = str(solver_cfg["model_name"])
    prompt_template = str(solver_cfg.get("prompt_template", "auto"))
    if prompt_template == "auto":
        prompt_template = detect_prompt_template(model_name)

    dataset = load_problem_split(config, args.split)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = Path(args.partial_output) if args.partial_output else None
    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text("", encoding="utf-8")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = _ensure_tokenizer_padding(
        AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=bool(solver_cfg.get("trust_remote_code", True)),
            local_files_only=bool(solver_cfg.get("local_files_only", True)),
        )
    )
    engine = LLM(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=bool(solver_cfg.get("trust_remote_code", True)),
        dtype=str(solver_cfg.get("torch_dtype", "float16")),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=max_model_len,
        max_num_seqs=int(args.max_num_seqs),
        skip_tokenizer_init=False,
        seed=random_seed,
    )

    temperature = float(solver_cfg.get("standard_cot_temperature", 0.7))
    do_sample = temperature > 1e-5
    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        top_p=float(solver_cfg.get("standard_cot_top_p", 0.8)) if do_sample else 1.0,
        repetition_penalty=float(solver_cfg.get("repetition_penalty", 1.0)),
        skip_special_tokens=True,
    )

    records: list[dict] = []
    completed = 0
    for batch in _chunks(dataset, max(1, int(args.batch_size))):
        prompts = []
        prompt_suffixes = []
        for problem in batch:
            user_prompt, suffix = _build_user_prompt(
                problem["question"],
                enable_thinking,
                prompt_style=str(args.prompt_style),
            )
            prompt_suffixes.append(suffix)
            prompts.append(
                _render_prompt_with_tokenizer(
                    tokenizer,
                    prompt_template=prompt_template,
                    system_prompt="You are a careful multiple-choice reasoning assistant.",
                    user_prompt=user_prompt,
                    enable_thinking=enable_thinking,
                )
            )
        started = perf_counter()
        outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
        batch_latency_ms = int((perf_counter() - started) * 1000)
        per_item_latency_ms = int(batch_latency_ms / max(1, len(outputs)))
        for problem, suffix, output in zip(batch, prompt_suffixes, outputs):
            best = output.outputs[0]
            text = best.text.strip()
            labels = _choice_labels(problem)
            prediction = _extract_mc_answer(text, labels)
            gold = str(problem.get("answer") or "").strip().upper()
            prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
            completion_tokens = len(getattr(best, "token_ids", []) or [])
            record = {
                "qid": problem["qid"],
                "prediction": prediction,
                "gold_answer": gold,
                "correct": prediction is not None and prediction.upper() == gold,
                "total_tokens": completion_tokens,
                "solver_tokens": completion_tokens,
                "verifier_tokens": 0,
                "latency_ms": per_item_latency_ms,
                "branches_used": 1,
                "stop_reason": "MC_DIRECT_COT",
                "actions": ["STANDARD_MC_DIRECT_COT"],
                "metadata": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "completion_head": text[:500],
                    "completion_tail": text[-500:],
                    "enable_thinking": enable_thinking,
                    "prompt_suffix": suffix,
                    "prompt_style": str(args.prompt_style),
                    "choice_labels": labels,
                    "contains_think_tag": "<think>" in text,
                    "contains_end_think_tag": "</think>" in text,
                    "backend": "vllm_batched_mc",
                    "batch_size": int(args.batch_size),
                    "max_model_len": max_model_len,
                    "max_new_tokens": max_new_tokens,
                    "max_num_seqs": int(args.max_num_seqs),
                    "random_seed": random_seed,
                },
            }
            if args.store_completion_text:
                record["metadata"]["completion_text"] = text
            records.append(record)
            completed += 1
            if partial_path is not None:
                with partial_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"method": "standard_direct_cot", **record}) + "\n")
            if args.progress_every > 0 and completed % args.progress_every == 0:
                print(
                    f"[{completed}/{len(dataset)}] id={record['qid']} pred={prediction} gold={gold} "
                    f"correct={record['correct']} tokens={completion_tokens}",
                    flush=True,
                )

    payload = {"standard_direct_cot": {"summary": summarize_records(records), "records": records}}
    dump_json(output_path, payload)
    print(f"wrote batched MC vLLM results to {output_path}")


if __name__ == "__main__":
    main()
