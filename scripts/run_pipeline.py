"""
Наскрізний конвеєр: дані -> ознаки -> навчання -> оцінювання -> графіки.

Запуск з кореня репозиторію:
    python scripts/run_pipeline.py
"""

import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import build_dataset
from src.features import FeatureExtractor
from src.features.stylometric import N_STYLOMETRIC, STYLOMETRIC_NAMES
from src.features.perplexity import N_PERPLEXITY, PERPLEXITY_NAMES
from src.model import StandardScaler
from src.train import train_model
from src.evaluate import (
    predict_scores, compute_metrics, plot_training, plot_roc,
    plot_confusion, plot_ablation, plot_feature_distribution,
)

from sklearn.linear_model import LogisticRegression


def main():
    t_start = time.time()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = yaml.safe_load(open(os.path.join(root, "config.yaml"), encoding="utf-8"))

    np.random.seed(cfg["seed"])
    figdir = os.path.join(root, cfg["paths"]["figures"])
    os.makedirs(figdir, exist_ok=True)

    # --- 1. Дані ---------------------------------------------------------
    print("\n[1/6] Формування набору даних...")
    data = build_dataset(cfg)

    # --- 2. Вилучення ознак ---------------------------------------------
    print("\n[2/6] Вилучення ознак (це найтриваліший етап)...")
    lm = cfg["language_model"]
    extractor = FeatureExtractor(
        lm_name=lm["name"], max_tokens=lm["max_tokens"],
        window_size=lm["window_size"], top_k=lm["top_k"],
    )

    feats = {}
    for part in ("train", "val", "test"):
        texts, labels = data[part]
        print(f"  розділ '{part}': {len(texts)} текстів")
        X = extractor.extract_batch(texts, verbose=True)
        feats[part] = (X, np.array(labels, dtype=np.float32))

    # --- 3. Нормалізація -------------------------------------------------
    scaler = StandardScaler().fit(feats["train"][0])
    Xtr = scaler.transform(feats["train"][0])
    Xva = scaler.transform(feats["val"][0])
    Xte = scaler.transform(feats["test"][0])
    ytr, yva, yte = feats["train"][1], feats["val"][1], feats["test"][1]

    np.savez(os.path.join(root, cfg["paths"]["features"]),
             Xtr=Xtr, Xva=Xva, Xte=Xte, ytr=ytr, yva=yva, yte=yte)

    # --- 4. Навчання повної моделі --------------------------------------
    print("\n[3/6] Навчання гібридної нейромережі (всі групи ознак)...")
    model, history = train_model(Xtr, ytr, Xva, yva, cfg)

    test_scores = predict_scores(model, Xte)
    metrics_full = compute_metrics(yte, test_scores)
    print(f"  тестова точність: {metrics_full['accuracy']:.4f} | "
          f"F1: {metrics_full['f1']:.4f} | AUC: {metrics_full['roc_auc']:.4f}")

    # --- 5. Абляційне дослідження та базові методи ----------------------
    print("\n[4/6] Абляційне дослідження груп ознак...")
    ablation_masks = {
        "Лише стилометрія":  {"stylometric": True,  "perplexity": False, "semantic": False},
        "Лише перплексія":   {"stylometric": False, "perplexity": True,  "semantic": False},
        "Лише семантика":    {"stylometric": False, "perplexity": False, "semantic": True},
        "Стилометрія+перплексія": {"stylometric": True, "perplexity": True, "semantic": False},
        "Повна модель (фузія)":   {"stylometric": True, "perplexity": True, "semantic": True},
    }
    ablation = {}
    for name, mask in ablation_masks.items():
        m, _ = train_model(Xtr, ytr, Xva, yva, cfg, branch_mask=mask)
        from src.train import _apply_mask
        dim_sem = Xte.shape[1] - N_STYLOMETRIC - N_PERPLEXITY
        sc = predict_scores(m, _apply_mask(Xte, mask, dim_sem))
        ablation[name] = compute_metrics(yte, sc)
        print(f"  {name:28s} -> acc={ablation[name]['accuracy']:.4f}, "
              f"F1={ablation[name]['f1']:.4f}")

    print("\n[5/6] Порівняння з базовими методами...")
    # Базовий метод 1: поріг за перплексією (підхід типу GPTZero).
    ppl_idx = N_STYLOMETRIC + PERPLEXITY_NAMES.index("perplexity")
    ppl_train = feats["train"][0][:, ppl_idx]
    best_thr, best_acc = 0.0, 0.0
    for thr in np.percentile(ppl_train, np.arange(5, 96, 2)):
        pred = (feats["train"][0][:, ppl_idx] < thr).astype(int)
        acc = (pred == ytr).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    ppl_pred = (feats["test"][0][:, ppl_idx] < best_thr).astype(int)
    baseline_ppl = compute_metrics(yte, ppl_pred.astype(float))

    # Базовий метод 2: логістична регресія на повному наборі ознак.
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xtr, ytr)
    lr_scores = lr.predict_proba(Xte)[:, 1]
    baseline_lr = compute_metrics(yte, lr_scores)

    baselines = {
        "Поріг за перплексією": baseline_ppl,
        "Логістична регресія": baseline_lr,
        "AuthorTrace (гібрид)": metrics_full,
    }
    for name, m in baselines.items():
        print(f"  {name:24s} -> acc={m['accuracy']:.4f}, F1={m['f1']:.4f}, "
              f"AUC={m['roc_auc']:.4f}")

    # --- 6. Збереження результатів та графіків --------------------------
    print("\n[6/6] Збереження моделі, метрик та ілюстрацій...")
    import torch
    torch.save(model.state_dict(), os.path.join(root, cfg["paths"]["model"]))
    json.dump(scaler.to_dict(),
              open(os.path.join(root, cfg["paths"]["scaler"]), "w"))

    results = {
        "dataset_source": data["source"],
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "feature_dim": int(Xtr.shape[1]),
        "semantic_dim": int(Xtr.shape[1] - N_STYLOMETRIC - N_PERPLEXITY),
        "full_model": metrics_full,
        "ablation": ablation,
        "baselines": baselines,
        "perplexity_threshold": float(best_thr),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    json.dump(results, open(os.path.join(root, cfg["paths"]["metrics"]), "w"),
              ensure_ascii=False, indent=2)

    plot_training(history, os.path.join(figdir, "fig_training.png"))
    plot_roc(yte, test_scores, os.path.join(figdir, "fig_roc.png"))
    plot_confusion(metrics_full["confusion_matrix"],
                   os.path.join(figdir, "fig_confusion.png"))
    plot_ablation(ablation, os.path.join(figdir, "fig_ablation.png"))

    # Розподіли двох найінформативніших стилометричних ознак.
    raw_tr = feats["train"][0]
    for fname in ("std_sentence_length", "hapax_ratio"):
        fi = STYLOMETRIC_NAMES.index(fname)
        h = raw_tr[ytr == 0][:, fi]
        a = raw_tr[ytr == 1][:, fi]
        plot_feature_distribution(
            h, a, fname, os.path.join(figdir, f"fig_dist_{fname}.png"))

    print(f"\nГотово за {results['runtime_seconds']} с. "
          f"Результати — у каталозі results/.")


if __name__ == "__main__":
    main()
