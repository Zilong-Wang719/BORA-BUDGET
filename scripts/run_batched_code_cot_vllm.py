from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bora.common import dump_json, load_jsonl
from bora.eval import summarize_records
from bora.llm import _ensure_tokenizer_padding, _render_prompt_with_tokenizer, detect_prompt_template


CODE_PROMPT = """You are given a Python programming task.

Write a correct Python function or functions that solve the task. Return only executable Python code in a single fenced ```python block. Do not include explanations.

Task:
{question}
"""


def _chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _build_user_prompt(question: str, enable_thinking: bool | None) -> tuple[str, str]:
    suffix = "/think" if enable_thinking is True else "/no_think"
    return f"{CODE_PROMPT.format(question=question).rstrip()}\n\n{suffix}", suffix


def _extract_code(text: str) -> str:
    if not text:
        return ""
    # Prefer code after the private thinking section.
    tail = text.split("</think>")[-1] if "</think>" in text else text
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", tail, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    any_fenced = re.findall(r"```\s*(.*?)```", tail, flags=re.DOTALL)
    if any_fenced:
        return any_fenced[-1].strip()
    return tail.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batched Qwen-style code generation with vLLM.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partial-output", default=None)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--store-completion-text", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if args.enable_thinking and args.disable_thinking:
        raise ValueError("Choose at most one of --enable-thinking and --disable-thinking.")
    enable_thinking = True if args.enable_thinking else False if args.disable_thinking else None

    dataset = load_jsonl(args.input)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = Path(args.partial_output) if args.partial_output else None
    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text("", encoding="utf-8")

    max_model_len = int(args.max_model_len or max(args.max_new_tokens + 2048, 4096))
    prompt_template = detect_prompt_template(args.model_name)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = _ensure_tokenizer_padding(
        AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True)
    )
    engine = LLM(
        model=args.model_name,
        tokenizer=args.model_name,
        trust_remote_code=True,
        dtype="float16",
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=max_model_len,
        max_num_seqs=int(args.max_num_seqs),
        skip_tokenizer_init=False,
        seed=int(args.random_seed),
    )
    do_sample = args.temperature > 1e-5
    sampling_params = SamplingParams(
        n=1,
        max_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature) if do_sample else 0.0,
        top_p=float(args.top_p) if do_sample else 1.0,
        skip_special_tokens=True,
    )

    records: list[dict] = []
    completed = 0
    for batch in _chunks(dataset, max(1, int(args.batch_size))):
        prompts = []
        suffixes = []
        for problem in batch:
            user_prompt, suffix = _build_user_prompt(problem["question"], enable_thinking)
            suffixes.append(suffix)
            prompts.append(
                _render_prompt_with_tokenizer(
                    tokenizer,
                    prompt_template=prompt_template,
                    system_prompt="You are a careful Python programming assistant.",
                    user_prompt=user_prompt,
                    enable_thinking=enable_thinking,
                )
            )
        started = perf_counter()
        outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
        per_item_latency_ms = int((perf_counter() - started) * 1000 / max(1, len(outputs)))
        for problem, suffix, output in zip(batch, suffixes, outputs):
            best = output.outputs[0]
            text = best.text.strip()
            code = _extract_code(text)
            prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
            completion_tokens = len(getattr(best, "token_ids", []) or [])
            metadata = dict(problem.get("metadata") or {})
            record = {
                "qid": problem["qid"],
                "prediction": code,
                "gold_answer": problem.get("answer") or "",
                "correct": False,
                "total_tokens": completion_tokens,
                "solver_tokens": completion_tokens,
                "verifier_tokens": 0,
                "latency_ms": per_item_latency_ms,
                "branches_used": 1,
                "stop_reason": "CODE_DIRECT_COT",
                "actions": ["CODE_DIRECT_COT"],
                "metadata": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "completion_head": text[:500],
                    "completion_tail": text[-500:],
                    "enable_thinking": enable_thinking,
                    "prompt_suffix": suffix,
                    "contains_think_tag": "<think>" in text,
                    "contains_end_think_tag": "</think>" in text,
                    "backend": "vllm_batched_code",
                    "batch_size": int(args.batch_size),
                    "max_model_len": max_model_len,
                    "max_new_tokens": int(args.max_new_tokens),
                    "max_num_seqs": int(args.max_num_seqs),
                    "random_seed": int(args.random_seed),
                    "tests": metadata.get("tests") or [],
                    "test_imports": metadata.get("test_imports") or [],
                    "task_prompt": metadata.get("prompt") or problem.get("question"),
                    "reference_code": metadata.get("reference_code") or "",
                },
            }
            if args.store_completion_text:
                record["metadata"]["completion_text"] = text
            records.append(record)
            completed += 1
            if partial_path is not None:
                with partial_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"method": "code_direct_cot", **record}) + "\n")
            if args.progress_every > 0 and completed % args.progress_every == 0:
                print(
                    f"[{completed}/{len(dataset)}] id={record['qid']} tokens={completion_tokens} "
                    f"code_chars={len(code)}",
                    flush=True,
                )

    payload = {"standard_direct_cot": {"summary": summarize_records(records), "records": records}}
    dump_json(output_path, payload)
    print(f"wrote batched code generations to {output_path}")


if __name__ == "__main__":
    main()
