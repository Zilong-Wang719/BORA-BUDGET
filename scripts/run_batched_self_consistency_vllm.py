from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.answer_extraction import extract_explicit_answer, extract_numeric_answer
from bora.baselines import STANDARD_COT_PROMPT
from bora.common import dump_json, is_correct, load_config, load_problem_split, normalize_answer
from bora.eval import summarize_records
from bora.llm import _ensure_tokenizer_padding, _render_prompt_with_tokenizer, detect_prompt_template


def _build_user_prompt(question: str, enable_thinking: bool | None) -> tuple[str, str]:
    suffix = "/think" if enable_thinking is True else "/no_think"
    user_prompt = f"{STANDARD_COT_PROMPT.format(question=question).rstrip()}\n\n{suffix}"
    return user_prompt, suffix


def _chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _extract_answer(text: str) -> str | None:
    prediction = extract_explicit_answer(text, prefer_numeric=True)
    if prediction is None:
        prediction = extract_numeric_answer(text)
    return prediction


def _majority_vote(predictions: list[str | None]) -> str | None:
    votes: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for idx, prediction in enumerate(predictions):
        normalized = normalize_answer(prediction)
        if normalized is None:
            continue
        votes[normalized] += 1
        first_seen.setdefault(normalized, idx)
    if not votes:
        return None
    return max(votes, key=lambda item: (votes[item], -first_seen[item]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batched vLLM standard-CoT self-consistency."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partial-output", default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--store-completion-text",
        action="store_true",
        help="Store full generations for each self-consistency sample for robust re-evaluation.",
    )
    parser.add_argument("--progress-every", type=int, default=16)
    args = parser.parse_args()

    if args.enable_thinking and args.disable_thinking:
        raise ValueError("Choose at most one of --enable-thinking and --disable-thinking.")

    loaded_config = load_config(args.config)
    random_seed = int(
        args.random_seed
        if args.random_seed is not None
        else loaded_config.get("random_seed", 0)
    )
    config = {**loaded_config, "mode": "eval", "random_seed": random_seed}
    llm_cfg = dict(config.get("llm", {}))
    solver_cfg = {**llm_cfg, **config.get("solver", {})}

    if args.enable_thinking:
        enable_thinking: bool | None = True
    elif args.disable_thinking:
        enable_thinking = False
    else:
        enable_thinking = solver_cfg.get(
            "standard_cot_enable_thinking", solver_cfg.get("enable_thinking")
        )

    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else solver_cfg.get("standard_cot_max_new_tokens", 1024)
    )
    max_model_len = int(
        args.max_model_len
        if args.max_model_len is not None
        else max(max_new_tokens + 1024, int(llm_cfg.get("max_context_tokens", 4096)))
    )
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
        n=max(1, int(args.num_samples)),
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
            user_prompt, suffix = _build_user_prompt(problem["question"], enable_thinking)
            prompt_suffixes.append(suffix)
            prompts.append(
                _render_prompt_with_tokenizer(
                    tokenizer,
                    prompt_template=prompt_template,
                    system_prompt="You are a careful math reasoner.",
                    user_prompt=user_prompt,
                    enable_thinking=enable_thinking,
                )
            )
        started = perf_counter()
        outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
        batch_latency_ms = int((perf_counter() - started) * 1000)
        per_item_latency_ms = int(batch_latency_ms / max(1, len(outputs)))
        for problem, suffix, output in zip(batch, prompt_suffixes, outputs):
            candidates = []
            predictions = []
            total_completion_tokens = 0
            for idx, candidate_output in enumerate(output.outputs):
                text = candidate_output.text.strip()
                prediction = _extract_answer(text)
                completion_tokens = len(getattr(candidate_output, "token_ids", []) or [])
                total_completion_tokens += completion_tokens
                predictions.append(prediction)
                candidate = {
                    "sample_index": idx,
                    "prediction": prediction,
                    "completion_tokens": completion_tokens,
                    "completion_head": text[:300],
                    "completion_tail": text[-500:],
                }
                if args.store_completion_text:
                    candidate["completion_text"] = text
                candidates.append(candidate)
            prediction = _majority_vote(predictions)
            prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
            votes = Counter(
                normalize_answer(item)
                for item in predictions
                if normalize_answer(item) is not None
            )
            record = {
                "qid": problem["qid"],
                "prediction": prediction,
                "gold_answer": problem.get("answer"),
                "correct": is_correct(prediction, problem.get("answer")),
                "total_tokens": total_completion_tokens,
                "solver_tokens": total_completion_tokens,
                "verifier_tokens": 0,
                "latency_ms": per_item_latency_ms,
                "branches_used": len(candidates),
                "stop_reason": "STANDARD_COT_SELF_CONSISTENCY",
                "actions": [f"STANDARD_DIRECT_COT_SC_{idx + 1}" for idx in range(len(candidates))],
                "metadata": {
                    "votes": dict(votes),
                    "candidates": candidates,
                    "prompt_tokens": prompt_tokens,
                    "enable_thinking": enable_thinking,
                    "prompt_suffix": suffix,
                    "backend": "vllm_batched",
                    "num_samples": int(args.num_samples),
                    "batch_size": int(args.batch_size),
                    "max_model_len": max_model_len,
                    "max_new_tokens": max_new_tokens,
                    "max_num_seqs": int(args.max_num_seqs),
                    "random_seed": random_seed,
                },
            }
            records.append(record)
            completed += 1
            if partial_path is not None:
                with partial_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"method": "self_consistency", **record}) + "\n")
            if args.progress_every > 0 and completed % args.progress_every == 0:
                print(
                    f"[{completed}/{len(dataset)}] id={record['qid']} "
                    f"correct={record['correct']} tokens={record['total_tokens']}",
                    flush=True,
                )

    payload = {"self_consistency": {"summary": summarize_records(records), "records": records}}
    dump_json(output_path, payload)
    print(f"wrote batched vLLM self-consistency results to {output_path}")


if __name__ == "__main__":
    main()
