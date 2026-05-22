"""
Модуль інтерпретації рішень та посегментного аналізу.

Реалізує три можливості, які відрізняють систему від типових детекторів:
  1. Оцінювання внеску кожної групи ознак (стилометрія / перплексія /
     семантика) у підсумковий вердикт методом абляції гілок мережі.
  2. Посегментний аналіз — визначення речень, найімовірніше згенерованих
     штучно (часткова детекція). Для неангломовних текстів сегменти
     обчислюються над оригіналом, а класифікація — над перекладом
     кожного сегмента, що дозволяє відобразити результат на тексті
     мовою оригіналу.
  3. Агрегацію посегментних результатів у фінальну ймовірність:
     значна частина «штучних» сегментів зрушує загальний вердикт навіть
     тоді, коли модель, бачачи весь текст одразу, поверталась би до
     людської мітки через ефект усереднення.

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
    """
    model.eval()
    x = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
    zero = torch.zeros_like(x)

    hs, hp, hm = model.branch_features(x)
    zs, zp, zm = model.branch_features(zero)

    def logit(a, b, c):
        return model.classifier(torch.cat([a, b, c], dim=1)).item()

    base = logit(zs, zp, zm)
    full = logit(hs, hp, hm)

    contrib = {
        "stylometric": logit(hs, zp, zm) - base,
        "perplexity": logit(zs, hp, zm) - base,
        "semantic": logit(zs, zp, hm) - base,
    }

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
    """Кількість речень у сегменті, щоб усього вийшло не більше `cap`."""
    if n_sentences <= base * cap:
        return base
    return max(base, (n_sentences + cap - 1) // cap)


def segment_analysis(detector, text, group_size=None):
    """
    Посегментний аналіз для одномовного (англійського) тексту.

    Повертає список словників із полями:
      * `text` — текст сегмента;
      * `translated_text` — None (для сумісності з варіантом перекладу);
      * `ai_probability` — ймовірність штучного походження.
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
        segments.append({
            "text": chunk,
            "translated_text": None,
            "ai_probability": prob,
        })
        if len(segments) >= _MAX_SEGMENTS:
            break
    return segments


def segment_analysis_translated(detector, original_text, translator,
                                group_size=None):
    """
    Посегментний аналіз для неангломовного тексту з перекладом.

    Сегменти формуються над ОРИГІНАЛЬНИМ текстом (щоб користувач бачив
    виділення на знайомій йому мові), але класифікація виконується над
    перекладом кожного сегмента англійською. У кожному словнику
    зберігаються обидві версії — оригінал у `text` та переклад у
    `translated_text`.
    """
    sentences = split_sentences(original_text)
    if not sentences:
        return []

    if group_size is None:
        group_size = _adaptive_group_size(len(sentences))

    segments = []
    for i in range(0, len(sentences), group_size):
        original_chunk = " ".join(sentences[i:i + group_size])
        if len(original_chunk.split()) < 8:
            continue
        translated_chunk = translator.translate(original_chunk)
        if not translated_chunk or not translated_chunk.strip():
            continue
        prob = detector.predict_proba(translated_chunk)
        segments.append({
            "text": original_chunk,
            "translated_text": translated_chunk,
            "ai_probability": prob,
        })
        if len(segments) >= _MAX_SEGMENTS:
            break
    return segments


def aggregate_verdict(global_p, segments, threshold):
    """
    Поєднує глобальну ймовірність і посегментну розбивку у фінальну
    ймовірність штучного походження тексту.

    Логіка: коли значна частина сегментів класифікована як штучна,
    вердикт зміщується відповідно — навіть якщо модель, що бачить
    одразу весь текст, відповіла би «людина» через ефект усереднення.

    :param global_p:  глобальна ймовірність від єдиного прогону моделі
    :param segments:  список словників з полем `ai_probability`
    :param threshold: поріг класифікації окремого сегмента
    :return: фінальна ймовірність штучного походження [0, 1]
    """
    if not segments or len(segments) < 3:
        # Для дуже коротких текстів покладаємось на глобальний результат.
        return float(global_p)

    seg_probs = np.asarray(
        [s["ai_probability"] for s in segments], dtype=np.float32
    )
    mean_p = float(seg_probs.mean())
    frac_above = float((seg_probs >= threshold).mean())

    # Кандидати на фінальне значення: глобальна оцінка та середня
    # посегментна ймовірність — система обере найсильніший сигнал.
    candidates = [float(global_p), mean_p]

    # Майоритарне правило: якщо ≥50% сегментів класифіковані як ШІ,
    # додаємо саму частку як кандидат (вона гарантовано буде ≥0.5).
    if frac_above >= 0.5:
        candidates.append(frac_above)
    # «Значна частка»: 30–50% — переконуємось, що вердикт відображає
    # суттєву присутність штучних фрагментів.
    elif frac_above >= 0.3:
        candidates.append(0.5 + (frac_above - 0.3))

    return float(np.clip(max(candidates), 0.0, 1.0))
