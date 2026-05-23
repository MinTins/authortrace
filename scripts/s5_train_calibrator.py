"""
Етап 5 конвеєра — навчання калібратора пост-обробки.

Калібратор — це окрема легка модель (логістична регресія), що приймає
на вхід ймовірність базової фузійної мережі та розширені стилометричні
ознаки і повертає скориговану ймовірність штучного походження. Він
використовується як альтернатива rule-based калібратору (rule-based
активний за замовчуванням). Зокрема, режим `hybrid` поєднує обидва
підходи.

Конвеєр:
  1. Завантажуємо базову модель AuthorTrace.
  2. На кожному калібрувальному тексті:
     a) обчислюємо розширені стилометричні ознаки на ОРИГІНАЛІ;
     b) запускаємо базову модель і отримуємо raw_p
        (з автоматичним перекладом для україномовного входу);
     c) визначаємо контекстні прапорці (мова, переклад, довжина).
  3. Збираємо матрицю входів калібратора.
  4. Навчаємо логістичну регресію.
  5. Зберігаємо калібратор у JSON та виводимо звіт про коефіцієнти.

Запуск з кореня репозиторію:
    python scripts/s5_train_calibrator.py
"""

import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import AuthorTraceDetector
from src.features.stylometric import (
    extract_stylometric,
    N_STYLOMETRIC_BASE,
)
from src.calibration_corpus import build_calibration_dataset
from src.calibrator import Calibrator
from src.translate import detect_language


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def logit(p, eps=1e-6):
    p = max(eps, min(1 - eps, float(p)))
    return float(np.log(p / (1 - p)))


def main():
    cfg = yaml.safe_load(
        open(os.path.join(ROOT, "config.yaml"), encoding="utf-8")
    )

    print("=" * 60)
    print("Навчання калібратора пост-обробки")
    print("=" * 60)

    print("\n[1/4] Завантаження базової моделі...")
    detector = AuthorTraceDetector(
        model_path=os.path.join(ROOT, cfg["paths"]["model"]),
        scaler_path=os.path.join(ROOT, cfg["paths"]["scaler"]),
        lm_name=cfg["language_model"]["name"],
        max_tokens=cfg["language_model"]["max_tokens"],
        window_size=cfg["language_model"]["window_size"],
        top_k=cfg["language_model"]["top_k"],
        mcfg=cfg["model"],
    )

    print("\n[2/4] Підготовка калібрувального корпусу...")
    texts, labels = build_calibration_dataset()
    n_human = sum(1 for l in labels if l == 0)
    n_ai = sum(1 for l in labels if l == 1)
    print(f"  Усього текстів: {len(texts)} (людина={n_human}, ШІ={n_ai})")

    print(
        "\n[3/4] Обчислення raw_p та розширених стилометричних ознак...\n"
        "      (для україномовних текстів потрібен переклад — це триває "
        "кілька хвилин)"
    )

    X_rows = []
    y_rows = []

    t_start = time.time()
    for i, (text, label) in enumerate(zip(texts, labels)):
        full_features = extract_stylometric(text)
        extended_features = full_features[N_STYLOMETRIC_BASE:]

        source_lang = detect_language(text)
        is_ukrainian = source_lang == "uk"
        raw_p = detector.predict_proba(text, translate=is_ukrainian)

        is_translated = is_ukrainian
        n_words = len(text.split())
        is_short = n_words < 80

        x = Calibrator.build_input(
            raw_logit=logit(raw_p),
            extended_features=extended_features,
            is_translated=is_translated,
            is_short=is_short,
            is_ukrainian=is_ukrainian,
        )
        X_rows.append(x)
        y_rows.append(label)

        elapsed = time.time() - t_start
        tag = "ШІ " if label == 1 else "Люд"
        preview = text[:50].replace("\n", " ")
        print(
            f"  [{i + 1:3d}/{len(texts)}] {tag} {source_lang} "
            f"raw_p={raw_p:.3f}  ({elapsed:5.1f}s)  «{preview}...»"
        )

    X = np.vstack(X_rows)
    y = np.array(y_rows)

    print("\n[4/4] Навчання логістичної регресії...")
    calibrator = Calibrator()
    calibrator.fit(X, y, l2=1.0, lr=0.5, n_iter=2000, verbose=True)

    preds = np.array([calibrator.predict_proba(x) for x in X])
    train_acc = float(((preds >= 0.5).astype(int) == y).mean())
    print(f"\nТочність на калібрувальному наборі: {train_acc:.4f}")

    print("\n" + calibrator.coefficient_report())

    output_path = os.path.join(ROOT, "results/calibrator.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    calibrator.save(output_path)
    print(f"\nКалібратор збережено: {output_path}")

    metrics = {
        "calibration_set_size": int(len(texts)),
        "n_human": int(n_human),
        "n_ai": int(n_ai),
        "training_accuracy": train_acc,
        "raw_vs_calibrated": [],
    }
    for i, (text, label, x) in enumerate(zip(texts, labels, X)):
        raw_p = float(1 / (1 + np.exp(-x[0])))
        cal_p = float(calibrator.predict_proba(x))
        metrics["raw_vs_calibrated"].append({
            "idx": i,
            "label": int(label),
            "raw_p": raw_p,
            "calibrated_p": cal_p,
            "raw_correct": int(int(raw_p >= 0.5) == label),
            "calibrated_correct": int(int(cal_p >= 0.5) == label),
            "preview": text[:80].replace("\n", " "),
        })

    raw_correct = sum(r["raw_correct"] for r in metrics["raw_vs_calibrated"])
    cal_correct = sum(r["calibrated_correct"] for r in metrics["raw_vs_calibrated"])
    metrics["raw_accuracy"] = raw_correct / len(texts)
    metrics["calibrated_accuracy"] = cal_correct / len(texts)

    print(f"\nЗведення на калібрувальному наборі:")
    print(f"  Raw (базова модель):       {raw_correct}/{len(texts)} "
          f"= {metrics['raw_accuracy']:.4f}")
    print(f"  Calibrated (з калібратором): {cal_correct}/{len(texts)} "
          f"= {metrics['calibrated_accuracy']:.4f}")

    metrics_path = os.path.join(ROOT, "results/calibration_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Метрики збережено: {metrics_path}")


if __name__ == "__main__":
    main()
