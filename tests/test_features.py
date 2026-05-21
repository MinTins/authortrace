"""
Модульні тести для перевірки коректності вилучення ознак.

Запуск з кореня репозиторію:
    python -m pytest tests/ -v
    або:  python tests/test_features.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.stylometric import extract_stylometric, N_STYLOMETRIC
from src.explain import split_sentences


HUMAN_SAMPLE = (
    "I waited ages for the train. It never came. Eventually I just walked, "
    "grumbling the whole way, and somehow that felt better than standing."
)
AI_SAMPLE = (
    "There are several important factors to consider. First, one must "
    "evaluate the context. Second, it is essential to weigh the options. "
    "In conclusion, a balanced approach is generally recommended."
)


def test_stylometric_shape():
    """Вектор стилометричних ознак має фіксовану розмірність."""
    vec = extract_stylometric(HUMAN_SAMPLE)
    assert vec.shape == (N_STYLOMETRIC,)
    assert vec.dtype == np.float32


def test_stylometric_empty():
    """Порожній текст не спричиняє помилки і дає нульовий вектор."""
    vec = extract_stylometric("")
    assert vec.shape == (N_STYLOMETRIC,)
    assert np.allclose(vec, 0.0)


def test_ttr_in_range():
    """Коефіцієнт лексичної різноманітності лежить у межах [0, 1]."""
    vec = extract_stylometric(HUMAN_SAMPLE)
    assert 0.0 <= vec[0] <= 1.0


def test_discourse_markers_detected():
    """У типовому згенерованому тексті частка дискурсивних маркерів вища."""
    h = extract_stylometric(HUMAN_SAMPLE)
    a = extract_stylometric(AI_SAMPLE)
    dm_index = 9  # discourse_marker_ratio
    assert a[dm_index] >= h[dm_index]


def test_sentence_split():
    """Поділ на речення відкидає надто короткі фрагменти."""
    sents = split_sentences(AI_SAMPLE)
    assert len(sents) >= 2
    assert all(len(s.split()) >= 6 for s in sents)


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} тестів пройдено")


if __name__ == "__main__":
    _run_all()
