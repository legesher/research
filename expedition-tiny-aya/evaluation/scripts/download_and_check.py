from datasets import load_dataset

BENCHMARKS = {
    "xnli": {
        "hf_id": "facebook/xnli",
        "langs": ["zh", "ur"],
        "preferred_splits": ["validation", "test", "train"],
    },
    "xstorycloze": {
        "hf_id": "juletxara/xstory_cloze",
        "langs": ["zh"],
        "preferred_splits": ["train", "validation", "test"],
    },
    "global_mmlu": {
        "hf_id": "CohereLabs/Global-MMLU",
        "langs": ["zh", "am"],
        "preferred_splits": ["test", "validation", "train"],
    },
}

def load_first_available_split(hf_id, lang, split_order):
    for split in split_order:
        try:
            ds = load_dataset(hf_id, lang, split=split)
            return split, ds
        except Exception:
            pass
    raise RuntimeError(f"Could not load any split for {hf_id} / {lang}")

for bench, meta in BENCHMARKS.items():
    for lang in meta["langs"]:
        split, ds = load_first_available_split(
            meta["hf_id"], lang, meta["preferred_splits"]
        )
        print(f"{bench} | {lang} | split={split} | rows={len(ds)} | cols={ds.column_names}")
        print(ds[0])
        print("-" * 80)