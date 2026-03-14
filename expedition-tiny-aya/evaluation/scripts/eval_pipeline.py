import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "CohereForAI/tiny-aya-base"

BENCHMARKS = {
    "xnli": "accuracy",
    "xstorycloze": "accuracy",
    "tydiqa": ["f1", "em"],
    "mmlu": "accuracy_per_subject",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks for one or more adapters."
    )
    parser.add_argument(
        "--adapter-path",
        action="append",
        required=True,
        help="Adapter path or HF Hub id. Repeat for batch mode.",
    )
    parser.add_argument("--language", required=True, help="Language code, e.g. zh, ur, am")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="Benchmarks to run: xnli xstorycloze tydiqa mmlu or all",
    )
    parser.add_argument("--output-file", required=True, help="JSON file for results")
    return parser.parse_args()


def normalize_benchmarks(values):
    if values[0].lower() == "all":
        return list(BENCHMARKS.keys())

    normalized = []
    for value in values:
        name = value.lower()
        if name not in BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {value}")
        normalized.append(name)
    return normalized


def load_model_with_adapter(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def load_dataset_split(path, config, splits):
    last_error = None
    for split in splits:
        try:
            return load_dataset(path, config, split=split)
        except Exception as error:
            last_error = error
    raise last_error


def move_to_model_device(inputs, model):
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in inputs.items()}


def score_choice(model, tokenizer, prompt, choice):
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    full_tokens = tokenizer(prompt + choice, return_tensors="pt", add_special_tokens=False)
    prompt_length = prompt_tokens["input_ids"].shape[1]
    full_tokens = move_to_model_device(full_tokens, model)

    with torch.no_grad():
        outputs = model(**full_tokens)
        logits = outputs.logits[:, :-1, :]
        targets = full_tokens["input_ids"][:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    choice_log_probs = token_log_probs[:, prompt_length - 1 :]
    return float(choice_log_probs.sum().item())


def generate_answer(model, tokenizer, prompt, max_new_tokens=32):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = move_to_model_device(inputs, model)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def normalize_text(text):
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return " ".join(text.split())


def exact_match(prediction, answers):
    prediction = normalize_text(prediction)
    for answer in answers:
        if prediction == normalize_text(answer):
            return 1.0
    return 0.0


def f1_score(prediction, answers):
    prediction_tokens = normalize_text(prediction).split()
    if not prediction_tokens:
        return 0.0

    best = 0.0
    for answer in answers:
        answer_tokens = normalize_text(answer).split()
        if not answer_tokens:
            continue

        overlap = 0
        remaining = list(answer_tokens)
        for token in prediction_tokens:
            if token in remaining:
                overlap += 1
                remaining.remove(token)

        if overlap == 0:
            continue

        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))

    return best


def get_answers(example):
    answers = example["answers"]
    if isinstance(answers, dict) and "text" in answers:
        return answers["text"]
    if isinstance(answers, list):
        return answers
    return [str(answers)]


def run_xnli(model, tokenizer, language):
    dataset = load_dataset("xnli", language, split="test")
    label_names = ["entailment", "neutral", "contradiction"]
    correct = 0

    for example in dataset:
        prompt = (
            "Read the premise and hypothesis. "
            "Decide whether the hypothesis is entailed by the premise, neutral, or contradictory.\n"
            f"Premise: {example['premise']}\n"
            f"Hypothesis: {example['hypothesis']}\n"
            "Answer:"
        )

        scores = [
            score_choice(model, tokenizer, prompt, f" {label}")
            for label in label_names
        ]
        prediction = int(torch.tensor(scores).argmax().item())
        correct += int(prediction == int(example["label"]))

    total = len(dataset)
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "num_samples": total, "language": language}


def run_xstorycloze(model, tokenizer, language):
    dataset = load_dataset_split(
        "juletxara/xstory_cloze", language, ["validation", "eval", "test"]
    )
    correct = 0

    for example in dataset:
        prompt = (
            "Read the story and choose the better ending.\n"
            f"{example['input_sentence_1']}\n"
            f"{example['input_sentence_2']}\n"
            f"{example['input_sentence_3']}\n"
            f"{example['input_sentence_4']}\n"
            "Best ending:"
        )

        ending_1_score = score_choice(
            model, tokenizer, prompt, f" {example['sentence_quiz1']}"
        )
        ending_2_score = score_choice(
            model, tokenizer, prompt, f" {example['sentence_quiz2']}"
        )

        prediction = 1 if ending_1_score > ending_2_score else 2
        correct += int(prediction == int(example["answer_right_ending"]))

    total = len(dataset)
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "num_samples": total, "language": language}


def run_tydiqa(model, tokenizer, language):
    dataset = load_dataset_split(
        "juletxara/tydiqa_xtreme", "secondary_task", ["validation", "test"]
    )
    dataset = dataset.filter(lambda example: example["language"] == language)

    em_total = 0.0
    f1_total = 0.0

    for example in dataset:
        prompt = (
            "Answer the question using only the context.\n"
            f"Context: {example['context']}\n"
            f"Question: {example['question']}\n"
            "Answer:"
        )

        prediction = generate_answer(model, tokenizer, prompt)
        answers = get_answers(example)
        em_total += exact_match(prediction, answers)
        f1_total += f1_score(prediction, answers)

    total = len(dataset)
    return {
        "f1": f1_total / total if total else 0.0,
        "em": em_total / total if total else 0.0,
        "num_samples": total,
        "language": language,
    }


def run_mmlu(model, tokenizer, language):
    if language != "en":
        raise ValueError("The default public MMLU dataset is English-only.")

    dataset = load_dataset("cais/mmlu", "all", split="test")
    subject_totals = {}
    subject_correct = {}
    option_labels = ["A", "B", "C", "D", "E", "F"]

    for example in dataset:
        prompt_lines = [f"Question: {example['question']}"]
        for index, choice in enumerate(example["choices"]):
            prompt_lines.append(f"{option_labels[index]}. {choice}")
        prompt_lines.append("Answer:")
        prompt = "\n".join(prompt_lines)

        scores = [
            score_choice(model, tokenizer, prompt, f" {option_labels[i]}")
            for i in range(len(example["choices"]))
        ]
        prediction_index = int(torch.tensor(scores).argmax().item())

        gold_answer = example["answer"]
        if isinstance(gold_answer, str):
            gold_index = option_labels.index(gold_answer)
        else:
            gold_index = int(gold_answer)

        subject = example.get("subject", "unknown")
        subject_totals[subject] = subject_totals.get(subject, 0) + 1
        subject_correct[subject] = subject_correct.get(subject, 0) + int(
            prediction_index == gold_index
        )

    accuracy_per_subject = {
        subject: subject_correct[subject] / subject_totals[subject]
        for subject in sorted(subject_totals)
    }

    return {
        "accuracy_per_subject": accuracy_per_subject,
        "num_samples": len(dataset),
        "language": language,
    }


def run_benchmark(model, tokenizer, benchmark_name, language):
    if benchmark_name == "xnli":
        return run_xnli(model, tokenizer, language)
    if benchmark_name == "xstorycloze":
        return run_xstorycloze(model, tokenizer, language)
    if benchmark_name == "tydiqa":
        return run_tydiqa(model, tokenizer, language)
    if benchmark_name == "mmlu":
        return run_mmlu(model, tokenizer, language)
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


def save_results(results, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")


def evaluate_adapter(adapter_path, language, benchmarks):
    print(f"Loading base model from HF: {BASE_MODEL}")
    print(f"Loading adapter: {adapter_path}")

    model, tokenizer = load_model_with_adapter(adapter_path)
    adapter_results = {
        "adapter_path": adapter_path,
        "language": language,
        "benchmarks": {},
        "timing_seconds": {},
    }

    for benchmark_name in benchmarks:
        print(f"Running {benchmark_name}...")
        start_time = time.perf_counter()
        adapter_results["benchmarks"][benchmark_name] = run_benchmark(
            model, tokenizer, benchmark_name, language
        )
        elapsed = time.perf_counter() - start_time
        adapter_results["timing_seconds"][benchmark_name] = elapsed
        print(f"Finished {benchmark_name} in {elapsed:.2f}s")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter_results


def main():
    args = parse_args()
    benchmarks = normalize_benchmarks(args.benchmarks)

    results = {
        "base_model": BASE_MODEL,
        "language": args.language,
        "benchmarks": benchmarks,
        "runs": [],
    }

    for adapter_path in args.adapter_path:
        results["runs"].append(
            evaluate_adapter(
                adapter_path=adapter_path,
                language=args.language,
                benchmarks=benchmarks,
            )
        )

    save_results(results, args.output_file)
    print(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
