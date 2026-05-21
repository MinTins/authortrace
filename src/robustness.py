"""
Модуль дослідження стійкості детектора до спроб обходу.

Імітує типовий сценарій ухилення: користувач злегка редагує згенерований
текст («олюднює» його), щоб уникнути виявлення. Перетворення спрямовані
на руйнування поверхневих сигналів — низької перплексії та характерних
дискурсивних маркерів — але зберігають семантичну структуру тексту.
"""

import random
import re

# Заміни формальних конекторів на розмовні відповідники.
_MARKER_REPLACEMENTS = {
    "however": "but", "therefore": "so", "moreover": "also",
    "furthermore": "and", "additionally": "also", "consequently": "so",
    "in conclusion": "so", "overall": "anyway", "thus": "so",
    "it is important to": "you should", "it is essential to": "you need to",
}

# Стягнення, типові для людського неформального письма.
_CONTRACTIONS = {
    "it is": "it's", "do not": "don't", "does not": "doesn't",
    "is not": "isn't", "are not": "aren't", "cannot": "can't",
    "will not": "won't", "they are": "they're", "you are": "you're",
    "that is": "that's", "there is": "there's",
}

# Невеликі помилки набору, що підвищують перплексію тексту.
_TYPO_WORDS = {"the": "teh", "and": "adn", "with": "wiht",
               "that": "taht", "this": "tihs", "have": "ahve"}


def perturb_text(text, intensity=0.7, seed=None):
    """
    Повертає «олюднену» версію згенерованого тексту.

    :param intensity: ймовірність застосування кожного перетворення
    :param seed: зерно генератора для відтворюваності
    """
    rng = random.Random(seed)
    low = text

    # 1. Заміна дискурсивних маркерів.
    for marker, repl in _MARKER_REPLACEMENTS.items():
        if rng.random() < intensity:
            low = re.sub(rf"\b{re.escape(marker)}\b", repl, low,
                         flags=re.IGNORECASE)

    # 2. Стягнення словосполучень.
    for full, short in _CONTRACTIONS.items():
        if rng.random() < intensity:
            low = re.sub(rf"\b{re.escape(full)}\b", short, low,
                         flags=re.IGNORECASE)

    # 3. Поодинокі помилки набору.
    words = low.split()
    for i, w in enumerate(words):
        key = w.lower().strip(".,;:!?")
        if key in _TYPO_WORDS and rng.random() < intensity * 0.4:
            words[i] = w.lower().replace(key, _TYPO_WORDS[key])
    low = " ".join(words)

    # 4. Перемішування порядку речень (руйнує плавність викладу).
    sentences = re.split(r"(?<=[.!?])\s+", low)
    if len(sentences) > 2 and rng.random() < intensity:
        head = sentences[0]
        rest = sentences[1:]
        rng.shuffle(rest)
        low = " ".join([head] + rest)

    return low


def perturb_batch(texts, intensity=0.7, seed=42):
    """Застосовує перетворення до списку текстів."""
    return [perturb_text(t, intensity=intensity, seed=seed + i)
            for i, t in enumerate(texts)]
