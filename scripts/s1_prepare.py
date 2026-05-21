"""Етап 1 конвеєра: формування набору даних та збереження у файл."""

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import build_dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    data = build_dataset(cfg)

    texts, labels, split = [], [], []
    for part in ("train", "val", "test"):
        t, l = data[part]
        texts.extend(t)
        labels.extend(l)
        split.extend([part] * len(t))

    out = {"source": data["source"], "texts": texts,
           "labels": labels, "split": split}
    json.dump(out, open(os.path.join(ROOT, "data", "dataset.json"), "w"),
              ensure_ascii=False)
    print(f"Збережено dataset.json: {len(texts)} текстів, джерело {data['source']}")


if __name__ == "__main__":
    main()
