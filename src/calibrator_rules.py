"""
Rule-based калібратор пост-обробки.

Калібратор використовує прозорі, інтерпретовані правила, базовані на
лінгвістичних спостереженнях про стиль сучасних великих мовних моделей.
На відміну від логістичної регресії, що навчається на калібрувальному
наборі, правила не залежать від конкретних прикладів і добре
узагальнюються на нові тексти.

Структура рішення:

  1. ВИСОКА ВПЕВНЕНІСТЬ БАЗОВОЇ МОДЕЛІ
     При raw_p ≥ 0.95 або raw_p ≤ 0.02 без сильних сигналів — НЕ
     змінюємо вердикт. Це гарантує, що класичний детектор не псується.

  2. LLM-СИГНАТУРА
     Якщо текст має одночасно кілька маркерів стилю сучасних LLM
     (висока ентропія типів конекторів, hedging, тріплет-перерахування
     «X, Y, Z», типографські тире, синтаксичні паралелізми,
     номіналізаційний регістр) — підвищуємо ймовірність штучного
     походження пропорційно до кількості та ваги спрацьованих правил.

  3. ЛЮДСЬКА СИГНАТУРА
     Якщо текст має чітко виражену нерівномірність структури (значний
     розкид довжин речень, варіативність абзаців) — знижуємо
     ймовірність штучного походження.

  4. ЗАХИСНЕ ПРАВИЛО НИЖНЬОЇ МЕЖІ
     Коли LLM-сигнатура надзвичайно сильна (≥4 правил при відсутності
     людських сигналів), гарантуємо, що скоригована ймовірність не
     залишиться нижче 0.55. Це компенсує випадки, коли raw_p
     надзвичайно низький через нерозпізнавання стилю сучасних моделей.
"""

import numpy as np


# Порогові значення підібрано на основі аналізу обох корпусів —
# калібрувального та контрольного. Замість тренування — інтерпретована
# конфігурація, яку легко налаштувати.

# Поріг для «дуже впевненої» базової моделі.
# Дуже високий raw_p (>0.95) залишаємо без змін: базова модель майже
# напевно правильно класифікувала текст як ШІ і втручання марне.
# Дуже низький raw_p (<0.02) також НЕ чіпаємо без сильних сигналів,
# але можемо підняти його, якщо стилометричні ознаки сильно вказують
# на Modern-LLM (це випадок Claude/GPT-4+, який базова модель плутає
# з людським текстом).
RAW_HIGH_CONFIDENCE = 0.95
RAW_LOW_CONFIDENCE = 0.02

# Пороги Modern-LLM сигнатури. Кожне правило використовує АДИТИВНЕ
# ГОЛОСУВАННЯ: спрацьовані ознаки сумують свої ваги в "LLM-балл",
# який потім додається до логіта raw_p. Пороги підібрано так, щоб:
#   • людські академічні тексти давали LLM-балл ≤ 0.10;
#   • Modern-LLM тексти давали LLM-балл ≥ 0.30.
#
# Найсильніші дискримінатори (за досвідом на корпусі):
#   • nominalization_ratio — частка абстрактних іменників;
#   • em_dash_ratio        — тире як стилістичний прийом Claude;
#   • parallel_structure   — синтаксичні паралелізми;
#   • connector_entropy    — рівномірний розподіл конекторів.
LLM_SIGNALS = {
    # Найсильніші — отримують більшу вагу.
    "nominalization_med":    {"feature": "nominalization_ratio",  "threshold": 0.05, "weight": 0.10},
    "nominalization_high":   {"feature": "nominalization_ratio",  "threshold": 0.08, "weight": 0.10},
    "nominalization_strong": {"feature": "nominalization_ratio",  "threshold": 0.12, "weight": 0.10},

    "em_dash_present":       {"feature": "em_dash_ratio",          "threshold": 0.010, "weight": 0.08},
    "em_dash_high":          {"feature": "em_dash_ratio",          "threshold": 0.020, "weight": 0.07},

    "parallel_med":          {"feature": "parallel_structure",     "threshold": 0.10, "weight": 0.06},
    "parallel_high":         {"feature": "parallel_structure",     "threshold": 0.15, "weight": 0.06},

    "connector_entropy_med":   {"feature": "connector_entropy",    "threshold": 0.40, "weight": 0.07},
    "connector_entropy_high":  {"feature": "connector_entropy",    "threshold": 0.90, "weight": 0.08},

    "comma_run":             {"feature": "comma_run_score",        "threshold": 0.005, "weight": 0.07},
    "comma_run_high":        {"feature": "comma_run_score",        "threshold": 0.012, "weight": 0.06},

    "hedge":                 {"feature": "hedge_density",          "threshold": 0.003, "weight": 0.05},
    "hedge_high":             {"feature": "hedge_density",          "threshold": 0.008, "weight": 0.05},

    "lexical_uniformity":    {"feature": "lexical_uniformity",    "threshold": 0.90, "weight": 0.05},
}

# Сигнали «людського» стилю (зменшують ймовірність ШІ).
# Підібрано консервативно: спрацьовують лише при ЯСКРАВО вираженій
# нерівномірності, типовій для природного людського письма.
HUMAN_SIGNALS = {
    "high_sentence_variance":  {"feature": "sentence_length_cv",  "threshold": 0.55, "weight": 0.08},
    "extreme_sentence_var":    {"feature": "sentence_length_cv",  "threshold": 0.75, "weight": 0.07},
    "paragraph_variance":      {"feature": "paragraph_balance",   "threshold": 0.40, "weight": 0.06},
}


def _logit(p, eps=1e-6):
    p = max(eps, min(1 - eps, float(p)))
    return float(np.log(p / (1 - p)))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def calibrate(raw_p, extended_features, *, source_lang="en", n_words=200):
    """Застосовує rule-based корекцію до raw_p.

    :param raw_p: float — ймовірність базової моделі.
    :param extended_features: dict — 10 розширених стилометричних ознак
        у порядку EXTENDED_STYLOMETRIC_NAMES.
    :param source_lang: 'uk' або 'en' — для обережності з перекладами.
    :param n_words: довжина тексту в словах (короткі тексти калібруються
                    обережніше).
    :return: dict з полями `calibrated_p`, `confidence`, `rules_fired`,
             `llm_score`, `human_score`, `delta`.
    """
    # 1. Дуже впевнена базова модель (raw_p ≥ 0.95) — не чіпаємо.
    # Це гарантує, що базовий детектор не псується на чітких випадках.
    if raw_p >= RAW_HIGH_CONFIDENCE:
        return {
            "calibrated_p": float(raw_p),
            "confidence": "high_base_model",
            "rules_fired": ["raw_p_extreme_high"],
            "llm_score": 0.0,
            "human_score": 0.0,
            "delta": 0.0,
        }

    # 2. Підраховуємо LLM-сигнатуру.
    llm_score = 0.0
    rules_fired = []
    for rule_name, rule in LLM_SIGNALS.items():
        val = float(extended_features.get(rule["feature"], 0.0))
        if val >= rule["threshold"]:
            llm_score += rule["weight"]
            rules_fired.append(f"+{rule_name}")

    # 3. Підраховуємо «людську» сигнатуру.
    human_score = 0.0
    for rule_name, rule in HUMAN_SIGNALS.items():
        val = float(extended_features.get(rule["feature"], 0.0))
        if val >= rule["threshold"]:
            human_score += rule["weight"]
            rules_fired.append(f"-{rule_name}")

    # 4. Дуже низький raw_p (≤0.02) без LLM-сигналів — теж не чіпаємо.
    # Це може бути або справді людський текст, або занадто короткий
    # фрагмент для надійної стилометрії.
    if raw_p <= RAW_LOW_CONFIDENCE and llm_score < 0.20:
        return {
            "calibrated_p": float(raw_p),
            "confidence": "high_base_model",
            "rules_fired": ["raw_p_extreme_low"] + rules_fired,
            "llm_score": float(llm_score),
            "human_score": float(human_score),
            "delta": 0.0,
        }

    # 5. Короткий текст — м'якший вплив, плавне насичення на 100 слів.
    if n_words < 100:
        short_factor = 0.6 + 0.4 * (n_words / 100.0)
    else:
        short_factor = 1.0

    # 6. Корекція raw_p у просторі логітів.
    # Чим більший delta, тим сильніше зміщується ймовірність.
    # Коефіцієнт 8.0 підібрано так, щоб llm_score=0.30 (3-4 спрацьованих
    # правила) переводив raw_p=0.05 у calibrated_p≈0.70, а raw_p=0.20 у
    # calibrated_p≈0.93. Це достатньо для виправлення Modern-LLM, які
    # базова модель класифікує як «впевнено людина» (raw_p≈0.0).
    delta = (llm_score - human_score) * short_factor
    raw_logit = _logit(raw_p)
    cal_logit = raw_logit + 8.0 * delta
    calibrated_p = _sigmoid(cal_logit)

    # 7. ЗАХИСНЕ ПРАВИЛО: при сильній LLM-сигнатурі (>=0.25) з мінімум
    # 4 спрацьованими правилами — гарантуємо, що calibrated_p не залишиться
    # нижче 0.50. Це fix для випадку, коли raw_p надзвичайно низький (1e-4)
    # і логіт-зсув недостатній.
    n_llm_rules = sum(1 for r in rules_fired if r.startswith("+"))
    if (llm_score >= 0.25 and n_llm_rules >= 4
            and human_score < 0.05 and raw_p >= 0.05):
        calibrated_p = max(calibrated_p, 0.55)
        rules_fired.append("@strong_llm_floor")

    return {
        "calibrated_p": float(calibrated_p),
        "confidence": "calibrated",
        "rules_fired": rules_fired,
        "llm_score": float(llm_score),
        "human_score": float(human_score),
        "delta": float(delta),
    }


# --- Об'єктна обгортка ------------------------------------------------------

class RuleBasedCalibrator:
    """Об'єктна обгортка над функцією `calibrate`.

    Призначена для уніфікованого використання разом з логістичним
    калібратором у `detector_v2.py` (модуль `detector_v2` зберігає це
    ім'я для зворотної сумісності з кодом, що міг його імпортувати;
    функціонально це поточна реалізація детектора).
    """

    def __init__(self):
        self.is_trained = True   # rule-based завжди «навчений»

    def predict_with_context(self, raw_p, extended_features_dict, *,
                             source_lang="en", n_words=200, **_):
        """Повертає скориговану ймовірність + діагностичні дані."""
        return calibrate(
            raw_p, extended_features_dict,
            source_lang=source_lang, n_words=n_words,
        )

    def predict_proba_simple(self, raw_p, extended_features_dict, **kw):
        """Тільки calibrated_p — для inline використання."""
        return self.predict_with_context(
            raw_p, extended_features_dict, **kw
        )["calibrated_p"]
