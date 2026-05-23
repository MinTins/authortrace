"""
Демонстраційний інтерфейс командного рядка для AuthorTrace.

Приклади запуску:
    # Аналіз тексту через аргумент:
    python scripts/demo_cli.py --text "Текст для перевірки..."

    # Аналіз тексту з файлу:
    python scripts/demo_cli.py --file document.txt

    # Україномовний текст (мова визначається автоматично, виконується переклад):
    python scripts/demo_cli.py --file ukrainian.txt

    # Перемикач режиму калібратора:
    python scripts/demo_cli.py --file doc.txt --mode rules    # за замовч.
    python scripts/demo_cli.py --file doc.txt --mode lr       # log. regression
    python scripts/demo_cli.py --file doc.txt --mode hybrid   # обидва
    python scripts/demo_cli.py --file doc.txt --mode none     # без калібратора

    # Детальний звіт зі стилометрією та правилами:
    python scripts/demo_cli.py --file doc.txt --verbose
"""

# Заглушення інфо-повідомлень — ПЕРЕД імпортом ML-бібліотек.
import os as _os
import warnings as _warnings
for _name, _value in {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    _os.environ.setdefault(_name, _value)
_warnings.filterwarnings("ignore", category=FutureWarning)
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector_v2 import AuthorTraceDetectorV2


def _bar(value, width=30):
    filled = int(round(value * width))
    return "#" * filled + "-" * (width - filled)


def main():
    parser = argparse.ArgumentParser(
        description="AuthorTrace — детектор штучних текстів"
    )
    parser.add_argument("--text", type=str, help="текст для аналізу")
    parser.add_argument("--file", type=str, help="файл з текстом для аналізу")
    parser.add_argument(
        "--mode", type=str, default="rules",
        choices=["rules", "lr", "hybrid", "none"],
        help="режим калібратора (rules за замовчуванням)",
    )
    parser.add_argument(
        "--no-auto-translate", action="store_true",
        help="вимкнути автоматичний переклад неангломовних текстів",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="детальний звіт зі стилометрією та правилами калібратора",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="поріг класифікації (за замовчуванням 0.5)",
    )
    args = parser.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        print("Введіть текст для аналізу (порожній рядок — завершення):")
        text = "\n".join(iter(input, ""))

    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    cfg = yaml.safe_load(
        open(os.path.join(root, "config.yaml"), encoding="utf-8")
    )

    calibrator_path = os.path.join(root, "results/calibrator.json")
    detector = AuthorTraceDetectorV2(
        model_path=os.path.join(root, cfg["paths"]["model"]),
        scaler_path=os.path.join(root, cfg["paths"]["scaler"]),
        calibrator_path=calibrator_path,
        lm_name=cfg["language_model"]["name"],
        max_tokens=cfg["language_model"]["max_tokens"],
        window_size=cfg["language_model"]["window_size"],
        top_k=cfg["language_model"]["top_k"],
        mcfg=cfg["model"],
        calibrator_mode=args.mode,
    )

    auto_translate = not args.no_auto_translate
    result = detector.analyze(
        text, threshold=args.threshold, auto_translate=auto_translate,
    )

    print("\n" + "=" * 64)
    print("  AUTHORTRACE — РЕЗУЛЬТАТ АНАЛІЗУ")
    print("=" * 64)

    tinfo = result.get("translation") or {}
    if tinfo.get("translated"):
        print(f"  Мова оригіналу:     {tinfo.get('source_language')}")
        print(f"  Аналіз базової моделі: над перекладом на англійську")
    elif tinfo.get("source_language") and tinfo.get("source_language") != "en":
        print(f"  Мова оригіналу:     {tinfo.get('source_language')} (без перекладу)")

    print(f"  Режим калібратора:  {args.mode}")
    print()
    print(f"  ВЕРДИКТ:            {result['verdict']}")
    print(f"  Імовірність ШІ:     {result['ai_probability'] * 100:5.1f}%  "
          f"[{_bar(result['ai_probability'])}]")
    print(f"  Впевненість:        {result['confidence'] * 100:5.1f}%")

    if result.get("calibrator_used"):
        raw_p = result["raw_probability"]
        print()
        print(f"  Базова модель:      {raw_p * 100:5.1f}%  "
              f"[{_bar(raw_p)}]  (raw_p)")
        print(f"  Метод калібратора:  {result.get('calibrator_method', '-')}")
        rf = result.get("rules_fired", [])
        if rf:
            print(f"  Спрацювало правил:  {len(rf)} → {', '.join(rf)}")

    print("-" * 64)
    print("  Внесок груп ознак у вердикт базової моделі:")
    for grp, share in sorted(result["feature_contributions"].items(),
                             key=lambda x: -x[1]):
        names = {
            "stylometric": "Стилометрія",
            "perplexity": "Перплексія",
            "semantic": "Семантика",
        }
        print(
            f"    {names[grp]:14s} {share:5.1f}%  [{_bar(share / 100)}]"
        )

    if args.verbose and "extended_features" in result:
        print("-" * 64)
        print("  Розширені стилометричні ознаки:")
        for name, val in result["extended_features"].items():
            print(f"    {name:<26s} {val:>7.4f}")

    print("-" * 64)
    n_seg = len(result.get("segments", []))
    print(f"  Посегментний аналіз ({n_seg} сегментів):")
    for i, seg in enumerate(result.get("segments", []), 1):
        mark = "ШІ" if seg["ai_probability"] >= args.threshold else "Люд."
        preview = seg["text"][:54].replace("\n", " ")
        print(
            f"    [{i:2d}] {seg['ai_probability'] * 100:5.1f}% "
            f"{mark:4s} | {preview}..."
        )
    print("=" * 64)


if __name__ == "__main__":
    main()
