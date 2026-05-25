from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_mc_question(question: str, labels: list[str], texts: list[str]) -> str:
    choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return (
        f"{question.strip()}\n\n"
        f"Choices:\n{choices}\n\n"
        "Choose the single best answer. Put only the final option letter in \\boxed{}."
    )


def _load_arc_challenge(limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        labels = [str(x).strip() for x in row["choices"]["label"]]
        texts = [str(x).strip() for x in row["choices"]["text"]]
        answer = str(row["answerKey"]).strip()
        if answer not in labels:
            continue
        rows.append(
            {
                "qid": f"arc_challenge_{row.get('id') or idx}",
                "question": _format_mc_question(str(row["question"]), labels, texts),
                "answer": answer,
                "dataset": "arc_challenge",
                "metadata": {
                    "source_id": row.get("id"),
                    "choice_labels": labels,
                    "choice_texts": texts,
                    "split": "test",
                },
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _load_commonsenseqa(limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    try:
        dataset = load_dataset("tau/commonsense_qa", split="validation")
    except Exception:
        dataset = load_dataset("commonsense_qa", split="validation")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        labels = [str(x).strip() for x in row["choices"]["label"]]
        texts = [str(x).strip() for x in row["choices"]["text"]]
        answer = str(row["answerKey"]).strip()
        if answer not in labels:
            continue
        rows.append(
            {
                "qid": f"commonsenseqa_{row.get('id') or idx}",
                "question": _format_mc_question(str(row["question"]), labels, texts),
                "answer": answer,
                "dataset": "commonsenseqa",
                "metadata": {
                    "source_id": row.get("id"),
                    "choice_labels": labels,
                    "choice_texts": texts,
                    "split": "validation",
                },
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _row_value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    lowered = {str(key).lower().replace(" ", "_"): value for key, value in row.items()}
    for name in names:
        key = name.lower().replace(" ", "_")
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def _load_gpqa(limit: int | None, *, subset: str = "gpqa_diamond", seed: int = 17) -> list[dict[str, Any]]:
    from datasets import DatasetDict, load_dataset

    candidates: list[tuple[str, str | None]] = [
        ("Idavidrein/gpqa", subset),
        ("Idavidrein/gpqa", "gpqa_main"),
        ("Idavidrein/gpqa", None),
        ("JingzeShi/gpqa_diamond", None),
        ("nichenshun/gpqa_diamond", None),
        ("dongboklee/GPQA-diamond", None),
        ("hendrydong/gpqa_diamond", None),
        ("talzoomanzoo/gpqa_diamond", None),
        ("Wanfq/gpqa", None),
        ("johnsonafool/gpqa", None),
    ]
    errors: list[str] = []
    dataset = None
    dataset_name = ""
    split_name = ""
    for name, config in candidates:
        try:
            loaded = load_dataset(name, config) if config else load_dataset(name)
            if isinstance(loaded, DatasetDict):
                split = "train" if "train" in loaded else next(iter(loaded.keys()))
                dataset = loaded[split]
                split_name = split
            else:
                dataset = loaded
                split_name = "unknown"
            dataset_name = f"{name}/{config or '-'}"
            break
        except Exception as exc:
            errors.append(f"{name}/{config or '-'}: {exc}")
    if dataset is None:
        raise RuntimeError("Could not load GPQA:\n" + "\n".join(errors))

    rows: list[dict[str, Any]] = []
    labels = ["A", "B", "C", "D"]
    for idx, row in enumerate(dataset):
        item = dict(row)
        question = _row_value(item, "Question", "question", "prompt", "problem")
        correct = _row_value(
            item,
            "Correct Answer",
            "correct_answer",
            "answer",
            "correct",
            "gold",
            "target",
        )
        incorrects = [
            _row_value(item, "Incorrect Answer 1", "incorrect_answer_1", "wrong_answer_1"),
            _row_value(item, "Incorrect Answer 2", "incorrect_answer_2", "wrong_answer_2"),
            _row_value(item, "Incorrect Answer 3", "incorrect_answer_3", "wrong_answer_3"),
        ]
        if item.get("incorrect_answers") and isinstance(item["incorrect_answers"], (list, tuple)):
            incorrects = list(item["incorrect_answers"])
        if item.get("choices") and isinstance(item["choices"], (list, tuple)):
            raw_choices = [str(choice).strip() for choice in item["choices"]]
            if correct is not None and str(correct).strip() in raw_choices:
                incorrects = [choice for choice in raw_choices if choice != str(correct).strip()][:3]
        # Some mirrors already provide answer choices as A/B/C/D columns and the
        # answer as a letter. Convert those into a correct text plus distractors.
        letter_choices = {
            label: _row_value(item, label, f"option_{label.lower()}", f"choice_{label.lower()}")
            for label in ["A", "B", "C", "D"]
        }
        if all(letter_choices.values()) and str(correct).strip().upper() in letter_choices:
            correct_label = str(correct).strip().upper()
            correct = letter_choices[correct_label]
            incorrects = [
                value for label, value in letter_choices.items() if label != correct_label
            ]
        if question is None or correct is None or any(ans is None for ans in incorrects):
            continue
        choices = [(str(correct).strip(), True)] + [(str(ans).strip(), False) for ans in incorrects]
        rng = random.Random(f"{seed}:{idx}:{question}")
        rng.shuffle(choices)
        texts = [text for text, _ in choices]
        answer = labels[[is_correct for _, is_correct in choices].index(True)]
        source_id = _row_value(item, "Record ID", "record_id", "id") or idx
        rows.append(
            {
                "qid": f"gpqa_{subset}_{source_id}",
                "question": _format_mc_question(str(question), labels, texts),
                "answer": answer,
                "dataset": subset,
                "metadata": {
                    "source_id": source_id,
                    "choice_labels": labels,
                    "choice_texts": texts,
                    "correct_answer_text": str(correct).strip(),
                    "split": split_name,
                    "hf_dataset": dataset_name,
                    "domain": _row_value(item, "High-level domain", "high_level_domain", "domain"),
                    "subdomain": _row_value(item, "Subdomain", "subdomain"),
                },
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["arc_challenge", "commonsenseqa", "gpqa_diamond", "gpqa_main"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.dataset == "arc_challenge":
        rows = _load_arc_challenge(args.limit)
    elif args.dataset == "commonsenseqa":
        rows = _load_commonsenseqa(args.limit)
    elif args.dataset == "gpqa_main":
        rows = _load_gpqa(args.limit, subset="gpqa_main", seed=int(args.seed))
    else:
        rows = _load_gpqa(args.limit, subset="gpqa_diamond", seed=int(args.seed))
    if not rows:
        raise SystemExit(f"No rows prepared for dataset={args.dataset}")
    _write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
