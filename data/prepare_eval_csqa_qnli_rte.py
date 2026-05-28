"""Prepare evaluation dataset combining CommonSenseQA, QNLI, and RTE in alpaca format."""

import json
from typing import Any

from datasets import load_dataset


LABEL_MAP: dict[int, str] = {0: "entailment", 1: "not_entailment"}

OUTPUT_PATH: str = "data/eval_csqa_qnli_rte.json"


def convert_commonsense_qa(example: dict[str, Any]) -> dict[str, str]:
    """Convert a single CommonSenseQA example to alpaca format."""
    question: str = example["question"]
    labels: list[str] = example["choices"]["label"]
    texts: list[str] = example["choices"]["text"]
    answer_key: str = example["answerKey"]

    choices_str = "\n".join(f"{label}) {text}" for label, text in zip(labels, texts))
    instruction = f"Question: {question}\nChoices:\n{choices_str}\nAnswer:"

    return {"instruction": instruction, "output": answer_key}


def convert_qnli(example: dict[str, Any]) -> dict[str, str]:
    """Convert a single QNLI example to alpaca format."""
    question: str = example["question"]
    sentence: str = example["sentence"]
    label: int = example["label"]

    instruction = (
        "Given the question and sentence, determine if the sentence answers the question.\n"
        f"Question: {question}\n"
        f"Sentence: {sentence}\n"
        "Answer:"
    )

    return {"instruction": instruction, "output": LABEL_MAP[label]}


def convert_rte(example: dict[str, Any]) -> dict[str, str]:
    """Convert a single RTE example to alpaca format."""
    premise: str = example["sentence1"]
    hypothesis: str = example["sentence2"]
    label: int = example["label"]

    instruction = (
        "Given the premise, determine if the hypothesis is entailed.\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:"
    )

    return {"instruction": instruction, "output": LABEL_MAP[label]}


def main() -> None:
    all_samples: list[dict[str, str]] = []

    # --- CommonSenseQA ---
    print("Loading CommonSenseQA validation split...")
    csqa = load_dataset("tau/commonsense_qa", split="validation")
    csqa_samples = [convert_commonsense_qa(ex) for ex in csqa]
    print(f"  CommonSenseQA: {len(csqa_samples)} samples")
    all_samples.extend(csqa_samples)

    # --- QNLI ---
    print("Loading QNLI validation split...")
    qnli = load_dataset("nyu-mll/glue", "qnli", split="validation")
    qnli_samples = [convert_qnli(ex) for ex in qnli]
    print(f"  QNLI: {len(qnli_samples)} samples")
    all_samples.extend(qnli_samples)

    # --- RTE ---
    print("Loading RTE validation split...")
    rte = load_dataset("nyu-mll/glue", "rte", split="validation")
    rte_samples = [convert_rte(ex) for ex in rte]
    print(f"  RTE: {len(rte_samples)} samples")
    all_samples.extend(rte_samples)

    # --- Save ---
    print(f"\nTotal samples: {len(all_samples)}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
