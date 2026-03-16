from datasets import get_dataset_config_names, load_dataset
import json
from pathlib import Path
from pprint import pprint

TARGET_LANGS = ["zh", "am", "ur"]

BENCHMARKS = {
    "xnli": {
        "hf_id": "facebook/xnli",
        "type": "classification",
    },
    "xstorycloze": {
        "hf_id": "juletxara/xstory_cloze",
        "type": "multiple_choice",
    },
    "tydiqa": {
        "hf_id": "SEACrowd/tydiqa",
        "type": "qa",
    },
    "global_mmlu": {
        "hf_id": "CohereLabs/Global-MMLU",
        "type": "multiple_choice",
    },
}

def try_load_split(hf_id, config_name=None):
    candidate_splits = ["validation", "test", "train", "dev"]

    for split in candidate_splits:
        try:
            ds = load_dataset(hf_id, config_name, split=split)
            first = ds[0] if len(ds) > 0 else None
            return {
                "split_loaded": split,
                "num_rows": len(ds),
                "columns": list(ds.column_names),
                "sample": first,
            }
        except Exception:
            continue

    return {
        "split_loaded": None,
        "num_rows": 0,
        "columns": [],
        "sample": None,
    }

def main():
    report = {}

    for bench_name, meta in BENCHMARKS.items():
        hf_id = meta["hf_id"]
        print(f"\n=== {bench_name} :: {hf_id} ===")

        entry = {
            "hf_id": hf_id,
            "available_configs": [],
            "target_lang_matches": {},
            "load_checks": {},
            "notes": [],
        }

        try:
            configs = get_dataset_config_names(hf_id)
            entry["available_configs"] = configs
        except Exception as e:
            entry["notes"].append(f"Could not fetch configs: {repr(e)}")
            configs = []

        # crude language matching
        for lang in TARGET_LANGS:
            matches = [c for c in configs if lang.lower() in c.lower()]
            entry["target_lang_matches"][lang] = matches

        # smoke-load matching configs
        checked = set()
        for lang in TARGET_LANGS:
            candidates = entry["target_lang_matches"][lang]

            # sometimes the config is exactly the language code
            if lang in configs:
                candidates = [lang] + candidates

            for cfg in candidates:
                if cfg in checked:
                    continue
                checked.add(cfg)

                try:
                    entry["load_checks"][cfg] = try_load_split(hf_id, cfg)
                except Exception as e:
                    entry["load_checks"][cfg] = {"error": repr(e)}

        # also try no-config loading for datasets that may not use per-language configs
        if not checked:
            try:
                entry["load_checks"]["__default__"] = try_load_split(hf_id, None)
            except Exception as e:
                entry["load_checks"]["__default__"] = {"error": repr(e)}

        report[bench_name] = entry
        pprint(entry)

    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir.parent / "results" / "baseline"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "benchmark_audit.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()