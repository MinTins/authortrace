"""
Модуль інтерпретації рішень та посегментного аналізу.

Реалізує дві можливості, які відрізняють систему від типових детекторів:
  1. Оцінювання внеску кожної групи ознак (стилометрія / перплексія /
     семантика) у підсумковий вердикт методом абляції гілок мережі.
  2. Посегментний аналіз — визначення речень, найімовірніше згенерованих
     штучно (часткова детекція).

Розмір сегмента підбирається адаптивно за довжиною тексту: коротким
текстам відповідають дрібні групи речень для детальної локалізації,
довгим — укрупнені, щоб кількість фрагментів залишалась осяжною як для
обчислень, так і для перегляду користувачем.
"""

import re

import numpy as np
import torch

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Верхня межа кількості сегментів у посегментному аналізі.
_MAX_SEGMENTS = 40


@torch.no_grad()
def branch_contributions(model, x_scaled):
    """
    Оцінює внесок кожної гілки мережі у відхилення логіта від базового
    рівня методом «залишити одну гілку».

    :param model: навчена FusionMLP
    :param x_scaled: нормалізований вектор ознак (1-D numpy)
    :return: словник {гілка: частка внеску у відсотках}
    """
    model.eval()
    x = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
    zero = torch.zeros_like(x)

    hs, hp, hm = model.branch_features(x)
    zs, zp, zm = model.branch_features(zero)

    def logit(a, b, c):
        return model.classifier(torch.cat([a, b, c], dim=1)).item()

    base = logit(zs, zp, zm)                  # базовий рівень
    full = logit(hs, hp, hm)                  # повний вердикт

    contrib = {
        "stylometric": logit(hs, zp, zm) - base,
        "perplexity": logit(zs, hp, zm) - base,
        "semantic": logit(zs, zp, hm) - base,
    }

    # Нормалізація внесків до відсотків від сумарного відхилення.
    total = sum(abs(v) for v in contrib.values()) + 1e-9
    shares = {k: 100.0 * abs(v) / total for k, v in contrib.items()}
    return {
        "base_logit": base,
        "full_logit": full,
        "raw": contrib,
        "shares": shares,
    }


def split_sentences(text, min_words=6):
    """Розбиває текст на речення, відкидаючи надто короткі фрагменти."""
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if len(p.split()) >= min_words]


def _adaptive_group_size(n_sentences, base=2, cap=_MAX_SEGMENTS):
    """
    Підбирає кількість речень у сегменті так, щоб усього вийшло не більше
    `cap` сегментів. Для короткого тексту повертає базовий розмір (2),
    для довгого — пропорційно більший.
    """
    if n_sentences <= base * cap:
        return base
    # ceil(n_sentences / cap)
    return max(base, (n_sentences + cap - 1) // cap)


def segment_analysis(detector, text, group_size=None):
    """
    Аналізує текст посегментно: групи з кількох речень класифікуються
    окремо, що дозволяє локалізувати штучно згенеровані ділянки.

    :param detector: екземпляр AuthorTraceDetector
    :param group_size: кількість речень в одному сегменті. Якщо None —
                       обирається адаптивно залежно від довжини тексту,
                       щоб кількість сегментів не перевищувала ліміту.
    :return: список словників {текст сегмента, ймовірність штучності}
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    if group_size is None:
        group_size = _adaptive_group_size(len(sentences))

    segments = []
    for i in range(0, len(sentences), group_size):
        chunk = " ".join(sentences[i:i + group_size])
        if len(chunk.split()) < 8:
            continue
        prob = detector.predict_proba(chunk)
        segments.append({"text": chunk, "ai_probability": prob})
        if len(segments) >= _MAX_SEGMENTS:
            break
    return segments
