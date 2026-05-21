"""
Демонстраційний інтерфейс командного рядка.

Приклади запуску:
    python scripts/demo_cli.py --text "Текст для перевірки..."
    python scripts/demo_cli.py --file document.txt
    python scripts/demo_cli.py --file ukrainian.txt --translate
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import AuthorTraceDetector
from src.translate import detect_language


def _bar(value, width=30):
    filled = int(round(value * width))
    return "#" * filled + "-" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="AuthorTrace — детектор штучних текстів")
    parser.add_argument("--text", type=str, help="текст для аналізу")
    parser.add_argument("--file", type=str, help="файл з текстом для аналізу")
    parser.add_argument(
        "--translate", action="store_true",
        help="перекладати неангломовний текст на англійську перед аналізом "
             "(потрібен пакет deep-translator)",
    )
    parser.add_argument(
        "--auto-translate", action="store_true",
        help="автоматично визначити мову; перекладати, якщо текст не "
             "англійською",
    )
    args = parser.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        print("Введіть текст для аналізу (порожній рядок — завершення):")
        text = "\n".join(iter(input, ""))

    # Виріши, чи робити переклад.
    translate = args.translate
    if args.auto_translate and not translate:
        translate = detect_language(text) != "en"

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = yaml.safe_load(open(os.path.join(root, "config.yaml"), encoding="utf-8"))

    detector = AuthorTraceDetector(
        model_path=os.path.join(root, cfg["paths"]["model"]),
        scaler_path=os.path.join(root, cfg["paths"]["scaler"]),
        lm_name=cfg["language_model"]["name"],
        max_tokens=cfg["language_model"]["max_tokens"],
        window_size=cfg["language_model"]["window_size"],
        top_k=cfg["language_model"]["top_k"],
        mcfg=cfg["model"],
    )

    result = detector.analyze(text, translate=translate)

    print("\n" + "=" * 56)
    print("  РЕЗУЛЬТАТ АНАЛІЗУ")
    print("=" * 56)

    tinfo = result.get("translation") or {}
    if tinfo.get("translated"):
        print(f"  Мова оригіналу:     {tinfo.get('source_language')}")
        print(f"  Аналізовано:        переклад на англійську")
    elif tinfo.get("source_language") and tinfo.get("source_language") != "en":
        print(f"  Мова оригіналу:     {tinfo.get('source_language')} "
              f"(БЕЗ перекладу — результат може бути ненадійним)")

    print(f"  Вердикт:            {result['verdict']}")
    print(f"  Імовірність ШІ:     {result['ai_probability'] * 100:5.1f}%  "
          f"[{_bar(result['ai_probability'])}]")
    print(f"  Впевненість моделі: {result['confidence'] * 100:5.1f}%")
    print("-" * 56)
    print("  Внесок груп ознак у вердикт:")
    for grp, share in sorted(result["feature_contributions"].items(),
                             key=lambda x: -x[1]):
        names = {"stylometric": "Стилометрія", "perplexity": "Перплексія",
                 "semantic": "Семантика"}
        print(f"    {names[grp]:14s} {share:5.1f}%  [{_bar(share / 100)}]")
    print("-" * 56)
    print(f"  Посегментний аналіз ({len(result['segments'])} сегментів):")
    for i, seg in enumerate(result["segments"], 1):
        mark = "ШІ" if seg["ai_probability"] >= 0.5 else "Люд."
        preview = seg["text"][:54].replace("\n", " ")
        print(f"    [{i:2d}] {seg['ai_probability'] * 100:5.1f}% {mark:4s} | {preview}...")
    print("=" * 56)


if __name__ == "__main__":
    main()
