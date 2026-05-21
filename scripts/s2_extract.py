"""
Етап 2 конвеєра: вилучення ознак з відновленням після переривання.

Скрипт обробляє за один запуск до CHUNK текстів і зберігає прогрес,
тому його можна викликати повторно до повного завершення.
"""

import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import FeatureExtractor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHUNK = 650  # кількість текстів за один запуск


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    data = json.load(open(os.path.join(ROOT, "data", "dataset.json"), encoding="utf-8"))
    texts = data["texts"]
    n = len(texts)

    feat_path = os.path.join(ROOT, "data", "features_flat.npy")
    prog_path = os.path.join(ROOT, "data", "extract_progress.json")

    if os.path.exists(feat_path) and os.path.exists(prog_path):
        feats = list(np.load(feat_path))
        done = json.load(open(prog_path))["done"]
    else:
        feats, done = [], 0

    if done >= n:
        print(f"Вилучення ознак уже завершено ({n} текстів).")
        return

    lm = cfg["language_model"]
    extractor = FeatureExtractor(
        lm_name=lm["name"], max_tokens=lm["max_tokens"],
        window_size=lm["window_size"], top_k=lm["top_k"],
    )

    end = min(done + CHUNK, n)
    print(f"Вилучення ознак: тексти {done}..{end} з {n}")
    for i in range(done, end):
        feats.append(extractor.extract(texts[i]))
        if (i + 1) % 100 == 0:
            print(f"  оброблено {i + 1}/{n}")

    np.save(feat_path, np.vstack(feats))
    json.dump({"done": end, "total": n}, open(prog_path, "w"))
    print(f"Збережено прогрес: {end}/{n}"
          + ("  — ЗАВЕРШЕНО" if end >= n else "  — потрібен ще запуск"))


if __name__ == "__main__":
    main()
