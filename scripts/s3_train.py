"""
Етап 3 конвеєра: навчання, оцінювання, абляційне дослідження, графіки.

Використовує ознаки, вилучені скриптом s2_extract.py.
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
from src.features.stylometric import N_STYLOMETRIC_BASE as N_STYLOMETRIC, STYLOMETRIC_NAMES
from src.features.perplexity import N_PERPLEXITY, PERPLEXITY_NAMES
from src.model import StandardScaler
from src.train import train_model, _apply_mask
from src.evaluate import (
    predict_scores, compute_metrics, plot_training, plot_roc,
    plot_confusion, plot_ablation, plot_feature_distribution,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    np.random.seed(cfg["seed"])
    figdir = os.path.join(ROOT, cfg["paths"]["figures"])

    data = json.load(open(os.path.join(ROOT, "data", "dataset.json"), encoding="utf-8"))
    X_all = np.load(os.path.join(ROOT, "data", "features_flat.npy"))
    y_all = np.array(data["labels"], dtype=np.float32)
    split = np.array(data["split"])

    Xtr_raw = X_all[split == "train"]
    Xva_raw = X_all[split == "val"]
    Xte_raw = X_all[split == "test"]
    ytr = y_all[split == "train"]
    yva = y_all[split == "val"]
    yte = y_all[split == "test"]

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xva, Xte = (scaler.transform(Xtr_raw),
                     scaler.transform(Xva_raw),
                     scaler.transform(Xte_raw))

    # --- Повна модель ----------------------------------------------------
    print("Навчання гібридної нейромережі (всі групи ознак)...")
    model, history = train_model(Xtr, ytr, Xva, yva, cfg)
    test_scores = predict_scores(model, Xte)
    metrics_full = compute_metrics(yte, test_scores)
    print(f"  тест: acc={metrics_full['accuracy']:.4f} "
          f"F1={metrics_full['f1']:.4f} AUC={metrics_full['roc_auc']:.4f}")

    # --- Абляційне дослідження ------------------------------------------
    print("Абляційне дослідження груп ознак...")
    dim_sem = Xte.shape[1] - N_STYLOMETRIC - N_PERPLEXITY
    masks = {
        "Лише стилометрія":  {"stylometric": True,  "perplexity": False, "semantic": False},
        "Лише перплексія":   {"stylometric": False, "perplexity": True,  "semantic": False},
        "Лише семантика":    {"stylometric": False, "perplexity": False, "semantic": True},
        "Стилометрія+перплексія": {"stylometric": True, "perplexity": True, "semantic": False},
        "Повна модель (фузія)":   {"stylometric": True, "perplexity": True, "semantic": True},
    }
    ablation = {}
    for name, mask in masks.items():
        m, _ = train_model(Xtr, ytr, Xva, yva, cfg, branch_mask=mask)
        sc = predict_scores(m, _apply_mask(Xte, mask, dim_sem))
        ablation[name] = compute_metrics(yte, sc)
        print(f"  {name:26s} acc={ablation[name]['accuracy']:.4f} "
              f"F1={ablation[name]['f1']:.4f}")

    # --- Базові методи ---------------------------------------------------
    print("Порівняння з базовими методами...")
    ppl_idx = N_STYLOMETRIC + PERPLEXITY_NAMES.index("perplexity")
    best_thr, best_acc = 0.0, 0.0
    for thr in np.percentile(Xtr_raw[:, ppl_idx], np.arange(5, 96, 2)):
        pred = (Xtr_raw[:, ppl_idx] < thr).astype(int)
        acc = (pred == ytr).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    ppl_pred = (Xte_raw[:, ppl_idx] < best_thr).astype(float)
    baseline_ppl = compute_metrics(yte, ppl_pred)

    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xtr, ytr)
    baseline_lr = compute_metrics(yte, lr.predict_proba(Xte)[:, 1])

    baselines = {
        "Поріг за перплексією": baseline_ppl,
        "Логістична регресія": baseline_lr,
        "AuthorTrace (гібрид)": metrics_full,
    }
    for name, m in baselines.items():
        print(f"  {name:24s} acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
              f"AUC={m['roc_auc']:.4f}")

    # --- Збереження ------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(ROOT, cfg["paths"]["model"]))
    json.dump(scaler.to_dict(),
              open(os.path.join(ROOT, cfg["paths"]["scaler"]), "w"))

    results = {
        "dataset_source": data["source"],
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "feature_dim": int(Xtr.shape[1]), "semantic_dim": int(dim_sem),
        "full_model": metrics_full, "ablation": ablation,
        "baselines": baselines, "perplexity_threshold": float(best_thr),
        "epochs_trained": len(history["train_loss"]),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(results, open(os.path.join(ROOT, cfg["paths"]["metrics"]), "w"),
              ensure_ascii=False, indent=2)

    plot_training(history, os.path.join(figdir, "fig_training.png"))
    plot_roc(yte, test_scores, os.path.join(figdir, "fig_roc.png"))
    plot_confusion(metrics_full["confusion_matrix"],
                   os.path.join(figdir, "fig_confusion.png"))
    plot_ablation(ablation, os.path.join(figdir, "fig_ablation.png"))
    for fname in ("std_sentence_length", "hapax_ratio", "perplexity", "burstiness"):
        if fname in STYLOMETRIC_NAMES:
            fi = STYLOMETRIC_NAMES.index(fname)
        else:
            fi = N_STYLOMETRIC + PERPLEXITY_NAMES.index(fname)
        plot_feature_distribution(
            Xtr_raw[ytr == 0][:, fi], Xtr_raw[ytr == 1][:, fi], fname,
            os.path.join(figdir, f"fig_dist_{fname}.png"))

    print(f"Готово за {results['runtime_seconds']} с.")


if __name__ == "__main__":
    main()
