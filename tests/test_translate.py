"""
Модульні тести для нової функціональності: визначення мови, чанкінг
довгих текстів для перекладача, посегментна обробка довгих текстів.

Запуск:
    python -m pytest tests/test_translate.py -v
    або:  python tests/test_translate.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.translate import detect_language, _split_for_translation, Translator
from src.explain import _adaptive_group_size, _MAX_SEGMENTS


# --- Визначення мови -------------------------------------------------------

def test_detect_english():
    assert detect_language("Just plain English here.") == "en"


def test_detect_ukrainian():
    assert detect_language("Це український текст для тесту.") == "uk"


def test_detect_empty_defaults_to_english():
    assert detect_language("") == "en"
    assert detect_language("   \n\t") == "en"


def test_detect_no_letters_defaults_to_english():
    assert detect_language("123 !!! 456") == "en"


def test_detect_mostly_cyrillic_with_some_latin():
    text = "Сьогодні я слухав Beatles і це було чудово."
    assert detect_language(text) == "uk"


def test_detect_mostly_latin_with_some_cyrillic():
    text = "Today I listened to а little music from the Beatles."
    assert detect_language(text) == "en"


# --- Чанкінг довгих текстів ------------------------------------------------

def test_chunking_short_text_stays_whole():
    text = "Це коротке речення."
    chunks = _split_for_translation(text, max_len=4500)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunking_respects_max_len():
    big = "Це довге речення. " * 500   # ~9000 chars
    chunks = _split_for_translation(big, max_len=4500)
    assert all(len(c) <= 4500 for c in chunks)
    assert len(chunks) >= 2


def test_chunking_handles_no_sentence_breaks():
    """Текст без розділових знаків — ріжемо за словами, не падаємо."""
    big = ("слово " * 2000).strip()
    chunks = _split_for_translation(big, max_len=500)
    assert all(len(c) <= 500 for c in chunks)
    # Жодних втрачених слів.
    assert sum(c.count("слово") for c in chunks) == 2000


# --- Перекладач: англійський passthrough -----------------------------------

def test_translator_skips_english():
    """Англомовний текст НЕ повинен викликати бекенд перекладача."""
    t = Translator(target="en")
    # Імітуємо: якщо бекенд викличеться — впаде, бо _impl не зайняти,
    # але translate() має повернути текст без виклику бекенду.
    out = t.translate("This is plain English. Nothing to translate.")
    assert out == "This is plain English. Nothing to translate."
    # Бекенд не повинен бути ініціалізованим.
    assert t._impl is None


def test_translator_needs_translation_logic():
    t = Translator(target="en")
    assert t.needs_translation("Привіт, друзі.") is True
    assert t.needs_translation("Hello, friends.") is False
    assert t.needs_translation("") is False
    assert t.needs_translation("   ") is False


# --- Адаптивний розмір сегментів для довгих текстів ------------------------

def test_adaptive_group_size_short():
    """Для невеликої кількості речень розмір сегмента — базовий (2)."""
    assert _adaptive_group_size(10, base=2, cap=_MAX_SEGMENTS) == 2
    assert _adaptive_group_size(40, base=2, cap=_MAX_SEGMENTS) == 2


def test_adaptive_group_size_long():
    """Для довгих текстів розмір сегмента росте, щоб уписатись у ліміт."""
    # 200 речень / cap 40 → потрібно ≥5 речень у сегменті
    gs = _adaptive_group_size(200, base=2, cap=40)
    assert gs >= 5
    # Перевіряємо, що загальна кількість сегментів не перевищує cap
    n_segments = (200 + gs - 1) // gs
    assert n_segments <= 40


def test_adaptive_group_size_returns_at_least_base():
    assert _adaptive_group_size(0) >= 2
    assert _adaptive_group_size(1) >= 2


# --- Агрегація фінального вердикту -----------------------------------------

from src.explain import aggregate_verdict


def _seg_list(probs):
    return [{"ai_probability": p} for p in probs]


def test_aggregate_short_falls_back_to_global():
    """Якщо сегментів обмаль — повертається саме глобальна оцінка."""
    assert aggregate_verdict(0.3, _seg_list([0.9, 0.9]), 0.5) == 0.3
    assert aggregate_verdict(0.85, [], 0.5) == 0.85


def test_aggregate_majority_ai_overrides_global_human():
    """Якщо більшість сегментів — ШІ, фінальний результат теж ШІ."""
    # 9 сегментів з ШІ, 6 з людиною; global вважає текст людським
    probs = [0.95] * 9 + [0.05] * 6
    final = aggregate_verdict(0.1, _seg_list(probs), 0.5)
    assert final >= 0.5, f"Очікувалось ≥0.5, отримано {final}"


def test_aggregate_majority_human_keeps_human():
    """Якщо більшість сегментів людські — вердикт лишається людським."""
    probs = [0.05] * 8 + [0.95] * 2
    final = aggregate_verdict(0.1, _seg_list(probs), 0.5)
    assert final < 0.5, f"Очікувалось <0.5, отримано {final}"


def test_aggregate_significant_partial_ai_triggers_ai():
    """30–50% штучних сегментів — мають зміщувати вердикт у бік ШІ."""
    # 4 з 10 сегментів — ШІ (40%)
    probs = [0.95] * 4 + [0.05] * 6
    final = aggregate_verdict(0.1, _seg_list(probs), 0.5)
    assert final >= 0.5, f"Очікувалось ≥0.5, отримано {final}"


def test_aggregate_isolated_ai_segment_not_flagged():
    """Один штучний сегмент серед людських — не призводить до ШІ-вердикту."""
    probs = [0.95] + [0.05] * 9   # 10%
    final = aggregate_verdict(0.05, _seg_list(probs), 0.5)
    assert final < 0.5


def test_aggregate_clipped_to_unit_interval():
    """Фінальне значення завжди в [0, 1]."""
    final = aggregate_verdict(1.5, _seg_list([0.99] * 10), 0.5)
    assert 0.0 <= final <= 1.0


# --- Поділ для перекладу: дуже довгі однопараграфні тексти ----------------

def test_chunking_handles_single_huge_paragraph_no_newlines():
    """
    Типовий випадок: курсова, експортована у TXT без перенесень рядка
    між абзацами. Розбиття все одно має дати чанки в межах ліміту.
    """
    sentence = "Це довгий приклад речення, написаного у курсовій роботі. "
    big = sentence * 200  # ≈ 11 000 символів, ОДИН блок без \\n
    chunks = _split_for_translation(big, max_len=3500)
    assert all(len(c) <= 3500 for c in chunks)
    assert len(chunks) >= 3


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")
            failed.append(t.__name__)
    print(f"\n{passed}/{len(tests)} тестів пройдено")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all())
