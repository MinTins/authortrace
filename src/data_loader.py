"""
Модуль формування навчального набору даних.

Основне джерело — корпус HC3 (Human ChatGPT Comparison Corpus): пари
відповідей на однакові запитання, написаних людиною та згенерованих
моделлю ChatGPT. Якщо доступу до мережі немає, використовується невеликий
вбудований резервний набір.
"""

import json
import os
import random

# Резервний набір на випадок відсутності доступу до HC3.
_FALLBACK_HUMAN = [
    "Honestly the bus was late again and I just stood there freezing, "
    "watching three of them go past in the wrong direction. By the time "
    "mine showed up I'd already missed the meeting, so that was fun.",
    "My grandmother used to make this soup with whatever was left in the "
    "fridge. It never tasted the same twice and somehow that was the point.",
]
_FALLBACK_AI = [
    "There are several key factors to consider when addressing this issue. "
    "First, it is important to understand the underlying context. Second, "
    "one should evaluate the available options carefully. In conclusion, a "
    "balanced approach is generally recommended.",
    "This topic encompasses a wide range of considerations. Overall, it is "
    "essential to weigh the advantages and disadvantages. Ultimately, the "
    "best outcome depends on the specific circumstances involved.",
]


def _clip_words(text, max_words):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def _accept(text, min_words, max_words):
    n = len(text.split())
    return n >= min_words


def load_hc3(samples_per_class, min_words, max_words, seed):
    """
    Завантажує корпус HC3 з HuggingFace Hub та формує збалансовані списки
    людських і згенерованих текстів.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    human, ai = [], []
    for r in rows:
        for h in r.get("human_answers", []):
            h = h.strip()
            if _accept(h, min_words, max_words):
                human.append(_clip_words(h, max_words))
        for a in r.get("chatgpt_answers", []):
            a = a.strip()
            if _accept(a, min_words, max_words):
                ai.append(_clip_words(a, max_words))

    rng = random.Random(seed)
    rng.shuffle(human)
    rng.shuffle(ai)
    return human[:samples_per_class], ai[:samples_per_class]


def build_dataset(cfg):
    """
    Формує набір даних згідно з конфігурацією та повертає словник зі
    списками текстів і мітками (0 — людина, 1 — штучний текст).
    """
    d = cfg["data"]
    seed = cfg["seed"]

    try:
        human, ai = load_hc3(
            d["samples_per_class"], d["min_words"], d["max_words"], seed
        )
        source = "HC3"
    except Exception as exc:  # резервний сценарій
        print(f"[увага] HC3 недоступний ({exc}); використано резервний набір")
        human = (_FALLBACK_HUMAN * d["samples_per_class"])[:d["samples_per_class"]]
        ai = (_FALLBACK_AI * d["samples_per_class"])[:d["samples_per_class"]]
        source = "fallback"

    texts = human + ai
    labels = [0] * len(human) + [1] * len(ai)

    # Перемішування зі збереженням відповідності міток.
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]

    # Поділ на train / val / test.
    n = len(texts)
    p_train, p_val, _ = d["split"]
    n_train = int(n * p_train)
    n_val = int(n * p_val)

    parts = {
        "source": source,
        "train": (texts[:n_train], labels[:n_train]),
        "val": (texts[n_train:n_train + n_val], labels[n_train:n_train + n_val]),
        "test": (texts[n_train + n_val:], labels[n_train + n_val:]),
    }
    print(f"Набір даних ({source}): "
          f"train={len(parts['train'][0])}, "
          f"val={len(parts['val'][0])}, "
          f"test={len(parts['test'][0])}")
    return parts
