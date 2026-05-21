"""
Модуль вилучення лінгвостатистичних та стилометричних ознак тексту.

Ознаки цієї групи є інтерпретованими та обчислюються без нейронної мережі.
Гіпотеза: штучно згенерований текст має нижчу варіативність структури
(довжини речень, лексичного складу), ніж текст, написаний людиною.
"""

import re
import math
from collections import Counter

import numpy as np

# Службові (функціональні) слова — класична основа стилометрії.
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "as", "by", "from", "that", "this", "these", "those", "it",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "not", "no", "so", "than", "then", "there", "here",
    "we", "you", "they", "he", "she", "i", "my", "your", "their", "its",
}

# Слова-маркери, типові для згенерованих відповідей (дискурсивні конектори).
DISCOURSE_MARKERS = {
    "however", "therefore", "moreover", "furthermore", "additionally",
    "overall", "consequently", "thus", "hence", "indeed", "ultimately",
    "importantly", "notably", "specifically", "generally",
}

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ']+")
_SENT_RE = re.compile(r"[.!?]+")
_VOWELS = "aeiouyаеєиіїоуюя"

# Назви ознак — використовуються для інтерпретації результатів.
STYLOMETRIC_NAMES = [
    "type_token_ratio",       # лексична різноманітність
    "hapax_ratio",            # частка слів, що трапились один раз
    "avg_word_length",        # середня довжина слова
    "avg_sentence_length",    # середня довжина речення (слів)
    "std_sentence_length",    # розкид довжин речень (варіативність)
    "sentence_length_entropy",# ентропія розподілу довжин речень
    "comma_ratio",            # щільність ком
    "punctuation_ratio",      # щільність розділових знаків
    "function_word_ratio",    # частка службових слів
    "discourse_marker_ratio", # частка дискурсивних маркерів
    "bigram_uniqueness",      # частка унікальних біграм
    "digit_ratio",            # частка цифрових символів
    "uppercase_ratio",        # частка великих літер
    "flesch_reading_ease",    # індекс легкості читання
    "long_word_ratio",        # частка довгих слів (понад 6 літер)
]

N_STYLOMETRIC = len(STYLOMETRIC_NAMES)


def _safe_div(a, b):
    return a / b if b else 0.0


def _entropy(values):
    """Ентропія Шеннона для дискретного розподілу значень."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12, 2)
    return ent


def _count_syllables(word):
    """Грубе оцінювання кількості складів за групами голосних."""
    word = word.lower()
    groups = re.findall(r"[%s]+" % _VOWELS, word)
    return max(1, len(groups))


def extract_stylometric(text):
    """
    Повертає numpy-вектор стилометричних ознак для одного тексту.

    :param text: вхідний рядок
    :return: np.ndarray розмірності N_STYLOMETRIC
    """
    words = _WORD_RE.findall(text)
    words_lower = [w.lower() for w in words]
    n_words = len(words)
    n_chars = max(1, len(text))

    # Поділ на речення.
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    sent_lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    sent_lengths = [s for s in sent_lengths if s > 0]

    if n_words == 0:
        return np.zeros(N_STYLOMETRIC, dtype=np.float32)

    # Лексична різноманітність.
    counts = Counter(words_lower)
    ttr = _safe_div(len(counts), n_words)
    hapax = _safe_div(sum(1 for c in counts.values() if c == 1), n_words)

    # Довжини слів і речень.
    avg_word_len = _safe_div(sum(len(w) for w in words), n_words)
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    std_sent_len = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    sent_entropy = _entropy(sent_lengths)

    # Пунктуація.
    comma_ratio = _safe_div(text.count(","), n_words)
    punct_ratio = _safe_div(sum(text.count(c) for c in ".,;:!?-"), n_words)

    # Службові слова та дискурсивні маркери.
    fw_ratio = _safe_div(sum(1 for w in words_lower if w in FUNCTION_WORDS), n_words)
    dm_ratio = _safe_div(sum(1 for w in words_lower if w in DISCOURSE_MARKERS), n_words)

    # Різноманітність біграм.
    bigrams = list(zip(words_lower, words_lower[1:]))
    bigram_uniq = _safe_div(len(set(bigrams)), len(bigrams)) if bigrams else 0.0

    # Символьні характеристики.
    digit_ratio = _safe_div(sum(c.isdigit() for c in text), n_chars)
    upper_ratio = _safe_div(sum(c.isupper() for c in text), n_chars)
    long_word_ratio = _safe_div(sum(1 for w in words if len(w) > 6), n_words)

    # Індекс легкості читання Флеша.
    n_sent = max(1, len(sent_lengths))
    n_syll = sum(_count_syllables(w) for w in words)
    flesch = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (n_syll / n_words)
    flesch = max(-50.0, min(120.0, flesch)) / 100.0  # масштабування

    return np.array([
        ttr, hapax, avg_word_len, avg_sent_len, std_sent_len, sent_entropy,
        comma_ratio, punct_ratio, fw_ratio, dm_ratio, bigram_uniq,
        digit_ratio, upper_ratio, flesch, long_word_ratio,
    ], dtype=np.float32)
