# --base_model
# --adapter_path
# --language
# --benchmarks
# --output_file

import argparse
import huggingface_hub
import torch
import json

BENCHMARKS = {
    "xnli": {
        "task_type": "classification",
        "metric": "accuracy",
        "runner": None,
    },
    "xstorycloze": {
        "task_type": "multiple_choice",
        "metric": "accuracy",
        "runner": None,
    },
    "tydiqa": {
        "task_type": "qa",
        "metric": ["f1", "em"],
        "runner": None,
    },
    "mmlu": {
        "task_type": "multiple_choice",
        "metric": "accuracy_per_subject",
        "runner": None,
    },
}

# Argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model") # maybe this comes from HF not local
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the adapter model")
    parser.add_argument("--language", type=str, required=True, help="Language for the evaluation")
    parser.add_argument("--benchmarks", nargs='+', required=True, help="List of benchmarks to evaluate on")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the evaluation results")
    return parser.parse_args()

def load_base_model(base_model):
    # TODO load from huggingface

def load_tokenizer(base_model):
    # TODO

def load_adapter(adapter_path):
    # TODO

def attach_adapter(base_model, adapters):
    # TODO
    # Option A: load adapter on top of base, easier and usually used for PEFT
    # Option B: merge and unload, usually for merged inference, harder

def run_benchmark(model, tokenizer, benchmark_name, language):
    runner = BENCHMARKS[benchmark_name]['runner']

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    args = parse_args()

    base_model = load_base_model(args.base_model)
    tokenizer = load_tokenizer(args.base_model)
    adapters = load_adapter(args.adapter_path)
    language = args.language
    model = attach_adapter(base_model, adapters)
    output_file = args.output_file

    results = {}

    for benchmark_name in args.benchmarks:
        if benchmark_name not in BENCHMARKS:
            print(f"Benchmark {benchmark_name} not supported, skipping.")
            continue
        results[benchmark_name] = run_benchmark(model, tokenizer, benchmark_name, language)

    return results


if __name__ == "__main__":
    main()
