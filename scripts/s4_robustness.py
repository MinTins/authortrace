"""
Етап 4 конвеєра: дослідження стійкості детектора до спроб обходу.

Згенеровані тексти тестової вибірки піддаються «олюдненню», після чого
вимірюється частка таких текстів, які детектор усе ще класифікує як
штучні (recall класу «штучний текст»). Порівнюються чотири підходи.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import FeatureExtractor
from src.features.stylometric import N_STYLOMETRIC
from src.features.perplexity import N_PERPLEXITY, PERPLEXITY_NAMES
from src.model import StandardScaler
from src.train import train_model, _apply_mask
from src.evaluate import predict_scores
from src.robustness import perturb_batch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def plot_robustness(clean, perturbed, path):
    """Стовпчикова діаграма recall до та після спроби обходу."""
    names = list(clean.keys())
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - w / 2, [clean[n] for n in names], w, color="0.3",
           edgecolor="black", label="Оригінальний текст")
    ax.bar(x + w / 2, [perturbed[n] for n in names], w, color="0.72",
           edgecolor="black", label="Після спроби обходу")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Recall класу «штучний текст»")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    np.random.seed(cfg["seed"])

    data = json.load(open(os.path.join(ROOT, "data", "dataset.json"), encoding="utf-8"))
    X_all = np.load(os.path.join(ROOT, "data", "features_flat.npy"))
    y_all = np.array(data["labels"], dtype=np.float32)
    split = np.array(data["split"])
    texts = np.array(data["texts"], dtype=object)

    Xtr_raw, ytr = X_all[split == "train"], y_all[split == "train"]
    Xva_raw, yva = X_all[split == "val"], y_all[split == "val"]
    Xte_raw, yte = X_all[split == "test"], y_all[split == "test"]
    test_texts = texts[split == "test"]

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xva = scaler.transform(Xtr_raw), scaler.transform(Xva_raw)
    dim_sem = Xtr.shape[1] - N_STYLOMETRIC - N_PERPLEXITY

    # Навчання моделей для порівняння.
    print("Навчання моделей для порівняння стійкості...")
    masks = {
        "Перплексія": {"stylometric": False, "perplexity": True, "semantic": False},
        "Семантика": {"stylometric": False, "perplexity": False, "semantic": True},
        "AuthorTrace (гібрид)": {"stylometric": True, "perplexity": True, "semantic": True},
    }
    models = {}
    for name, mask in masks.items():
        models[name], _ = train_model(Xtr, ytr, Xva, yva, cfg, branch_mask=mask)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)

    ppl_idx = N_STYLOMETRIC + PERPLEXITY_NAMES.index("perplexity")
    best_thr, best_acc = 0.0, 0.0
    for thr in np.percentile(Xtr_raw[:, ppl_idx], np.arange(5, 96, 2)):
        acc = ((Xtr_raw[:, ppl_idx] < thr).astype(int) == ytr).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    # Згенеровані тексти тестової вибірки та їхні «олюднені» версії.
    ai_mask = yte == 1
    ai_texts = list(test_texts[ai_mask])
    ai_perturbed = perturb_batch(ai_texts, intensity=0.7, seed=cfg["seed"])

    extractor = FeatureExtractor(
        lm_name=cfg["language_model"]["name"],
        max_tokens=cfg["language_model"]["max_tokens"],
        window_size=cfg["language_model"]["window_size"],
        top_k=cfg["language_model"]["top_k"],
    )

    def recall_on(text_list):
        """Частка текстів зі списку, класифікованих як штучні."""
        Xr = extractor.extract_batch(text_list, verbose=False)
        Xs = scaler.transform(Xr)
        res = {}
        res["Поріг за перплексією"] = float(
            np.mean(Xr[:, ppl_idx] < best_thr))
        for name, mask in masks.items():
            sc = predict_scores(models[name], _apply_mask(Xs, mask, dim_sem))
            res[name] = float(np.mean(sc >= 0.5))
        res["Логістична регресія"] = float(
            np.mean(lr.predict_proba(Xs)[:, 1] >= 0.5))
        return res

    print("Оцінювання на оригінальних згенерованих текстах...")
    clean = recall_on(ai_texts)
    print("Оцінювання на текстах після спроби обходу...")
    perturbed = recall_on(ai_perturbed)

    order = ["Поріг за перплексією", "Перплексія", "Семантика",
             "Логістична регресія", "AuthorTrace (гібрид)"]
    clean = {k: clean[k] for k in order}
    perturbed = {k: perturbed[k] for k in order}

    print("\n  Метод                     recall(чистий)  recall(обхід)  Δ")
    for k in order:
        d = clean[k] - perturbed[k]
        print(f"  {k:26s}  {clean[k]:.3f}          {perturbed[k]:.3f}"
              f"        -{d:.3f}")

    out = {"clean_recall": clean, "perturbed_recall": perturbed,
           "n_ai_texts": len(ai_texts),
           "runtime_seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(ROOT, "results", "robustness.json"), "w"),
              ensure_ascii=False, indent=2)
    plot_robustness(clean, perturbed,
                    os.path.join(ROOT, "results", "fig_robustness.png"))
    print(f"\nГотово за {out['runtime_seconds']} с.")


if __name__ == "__main__":
    main()
