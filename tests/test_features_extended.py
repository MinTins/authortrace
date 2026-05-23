"""Модульні тести розширеної стилометрії та калібраторів."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.stylometric import (
    extract_stylometric,
    STYLOMETRIC_NAMES,
    EXTENDED_STYLOMETRIC_NAMES,
    N_STYLOMETRIC,
    N_STYLOMETRIC_BASE,
    N_STYLOMETRIC_EXTENDED,
)
from src.calibrator_rules import calibrate, RuleBasedCalibrator


# --- Стилометрія --------------------------------------------------------

def test_stylometric_returns_full_vector():
    """Повний вектор містить базові + розширені ознаки."""
    feats = extract_stylometric("This is a simple test sentence.")
    assert feats.shape == (N_STYLOMETRIC,)
    assert N_STYLOMETRIC == N_STYLOMETRIC_BASE + N_STYLOMETRIC_EXTENDED
    assert len(STYLOMETRIC_NAMES) == N_STYLOMETRIC
    assert len(EXTENDED_STYLOMETRIC_NAMES) == N_STYLOMETRIC_EXTENDED


def test_base_dimensions_match_network():
    """Базова підгрупа має фіксовану розмірність 15 (вхід фузійної мережі)."""
    assert N_STYLOMETRIC_BASE == 15
    assert N_STYLOMETRIC_EXTENDED == 10


def test_empty_text_returns_zeros():
    """Порожній текст не має ламати функцію."""
    feats = extract_stylometric("")
    assert feats.shape == (N_STYLOMETRIC,)
    assert np.all(feats == 0)


def test_em_dash_detection():
    """em_dash_ratio зростає на текстах з типографськими тире."""
    no_dash = "This is a simple text without any typographic dashes."
    with_dash = "This is a text — with typographic dashes — used as style."
    f_no = extract_stylometric(no_dash)
    f_with = extract_stylometric(with_dash)
    em_idx = STYLOMETRIC_NAMES.index("em_dash_ratio")
    assert f_no[em_idx] == 0.0
    assert f_with[em_idx] > 0.0


def test_comma_triplet_detection():
    """comma_run_score знаходить тріплет-перерахування."""
    no_triplet = "This is a simple sentence with one comma."
    with_triplet = "Three things matter: speed, reliability, simplicity."
    f_no = extract_stylometric(no_triplet)
    f_with = extract_stylometric(with_triplet)
    cr_idx = STYLOMETRIC_NAMES.index("comma_run_score")
    assert f_no[cr_idx] == 0.0
    assert f_with[cr_idx] > 0.0


def test_nominalization_bilingual():
    """nominalization_ratio працює і для англійської, і для української."""
    en_text = ("The implementation of the optimization required modification "
               "of the documentation.")
    uk_text = "Реалізація оптимізації потребувала модифікації документації."
    f_en = extract_stylometric(en_text)
    f_uk = extract_stylometric(uk_text)
    nom_idx = STYLOMETRIC_NAMES.index("nominalization_ratio")
    assert f_en[nom_idx] >= 0.15
    assert f_uk[nom_idx] >= 0.15


def test_hedge_density_uk():
    """hedge_density розпізнає українські «виважувальні» слова."""
    no_hedge = "Це працює. Це швидко."
    with_hedge = "Це може працювати. Зазвичай швидко. Як правило, надійно."
    f_no = extract_stylometric(no_hedge)
    f_with = extract_stylometric(with_hedge)
    hd_idx = STYLOMETRIC_NAMES.index("hedge_density")
    assert f_with[hd_idx] > f_no[hd_idx]


# --- Калібратор правил --------------------------------------------------

def _zero_extended():
    """Словник з усіма нульовими розширеними ознаками."""
    return {name: 0.0 for name in EXTENDED_STYLOMETRIC_NAMES}


def test_calibrate_extreme_raw_p_unchanged():
    """raw_p >= 0.95 НЕ змінюється навіть при відсутності сигналів."""
    result = calibrate(0.99, _zero_extended())
    assert result["calibrated_p"] == 0.99


def test_calibrate_low_raw_p_no_signals():
    """raw_p <= 0.02 без сильних сигналів — теж не змінюється."""
    result = calibrate(0.001, _zero_extended())
    assert result["calibrated_p"] == 0.001


def test_calibrate_low_raw_p_strong_llm_signals():
    """Низький raw_p, але сильні LLM-ознаки — піднімається вище 0.5."""
    ext = _zero_extended()
    ext["nominalization_ratio"] = 0.15
    ext["em_dash_ratio"] = 0.025
    ext["parallel_structure"] = 0.20
    ext["comma_run_score"] = 0.015
    result = calibrate(0.001, ext)
    # Має спрацювати @strong_llm_floor.
    assert result["calibrated_p"] >= 0.50


def test_calibrate_preserves_confident_ai_verdict():
    """Текст з raw_p=0.99 та LLM-сигналами залишається впевнено «ШІ»."""
    ext = _zero_extended()
    ext["nominalization_ratio"] = 0.10
    ext["em_dash_ratio"] = 0.012
    result = calibrate(0.99, ext)
    assert result["calibrated_p"] >= 0.95


def test_rules_calibrator_class():
    """RuleBasedCalibrator drop-in інтерфейс."""
    cal = RuleBasedCalibrator()
    assert cal.is_trained
    p = cal.predict_proba_simple(
        0.5, _zero_extended(), source_lang="en", n_words=200,
    )
    assert 0.0 <= p <= 1.0


# --- Запуск ------------------------------------------------------------

def run_all():
    tests = [
        test_stylometric_returns_full_vector,
        test_base_dimensions_match_network,
        test_empty_text_returns_zeros,
        test_em_dash_detection,
        test_comma_triplet_detection,
        test_nominalization_bilingual,
        test_hedge_density_uk,
        test_calibrate_extreme_raw_p_unchanged,
        test_calibrate_low_raw_p_no_signals,
        test_calibrate_low_raw_p_strong_llm_signals,
        test_calibrate_preserves_confident_ai_verdict,
        test_rules_calibrator_class,
    ]
    passed = 0
    failed = []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  ✗ {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, f"EXCEPTION: {e}"))
            print(f"  ✗ {t.__name__}: EXCEPTION {e}")

    print()
    print(f"Підсумок: {passed}/{len(tests)} тестів пройдено")
    if failed:
        print("Помилки:")
        for n, msg in failed:
            print(f"  {n}: {msg}")
        sys.exit(1)
    print("Усі тести пройдено! ✓")


if __name__ == "__main__":
    run_all()
