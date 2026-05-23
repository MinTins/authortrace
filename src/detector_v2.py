"""
AuthorTrace — повний детектор з калібратором пост-обробки.

Цей модуль є основною точкою входу для аналізу текстів. Він поєднує
базову фузійну нейромережу (стилометрія + перплексія + семантика) з
шаром калібрування, що враховує стиль сучасних великих мовних моделей
та анти-зсуви від машинного перекладу.

Калібратор має чотири режими, доступні через параметр `calibrator_mode`:

  * 'rules'  — rule-based калібратор з прозорими правилами,
               не вимагає тренування. Залишає raw_p без змін, коли
               базова модель дуже впевнена або стилометричні сигнали
               слабкі. Це режим за замовчуванням.

  * 'lr'     — логістична регресія, навчена на калібрувальному наборі.
               Підходить для сценаріїв, де розподіл цільових текстів
               відповідає калібрувальному корпусу.

  * 'hybrid' — поєднання обох: якщо обидва калібратори згодні щодо
               класу, повертається їх середнє; якщо ні — rule-based.

  * 'none'   — повне вимкнення калібрування; повертає raw_p
               без модифікацій (поведінка лише базової фузійної мережі).

Архітектура:

                ┌─────────────────────────────────────┐
   текст ──────▶│ Базова фузійна нейромережа          │──┐
                │ (15 стилометр. + 8 перпл. + 768 сем.)│  │ raw_p
                └─────────────────────────────────────┘  │
                                                          ▼
                ┌─────────────────────────────────────┐  ┌──────────────┐
   текст ──────▶│ Розширений стилометричний модуль    │─▶│ Калібратор   │
                │ (10 додаткових ознак)               │  │ (rules / lr) │
                └─────────────────────────────────────┘  └──────────────┘
                                                          │
                                                          ▼
                                                    calibrated_p

Розширені стилометричні ознаки обчислюються над оригіналом тексту
(без машинного перекладу), що зберігає стилістичні характеристики мови
оригіналу та забезпечує надійну роботу для україномовних входів.
"""

import os

import numpy as np

from .detector import AuthorTraceDetector
from .features.stylometric import (
    extract_stylometric,
    EXTENDED_STYLOMETRIC_NAMES,
    N_STYLOMETRIC_BASE,
)
from .calibrator import Calibrator
from .calibrator_rules import RuleBasedCalibrator, calibrate as rule_calibrate
from .translate import detect_language


def _logit(p, eps=1e-6):
    p = max(eps, min(1 - eps, float(p)))
    return float(np.log(p / (1 - p)))


class AuthorTraceDetectorV2:
    """Повний детектор: базова мережа + калібратор пост-обробки.

    Назву збережено для зворотної сумісності із зовнішнім кодом,
    що міг імпортувати її явно. Функціонально — це поточна реалізація
    детектора з калібруванням.
    """

    VALID_MODES = ("rules", "lr", "hybrid", "none")

    def __init__(self, model_path, scaler_path, calibrator_path=None,
                 lm_name="distilgpt2", max_tokens=220, window_size=40,
                 top_k=10, mcfg=None, calibrator_mode="rules"):
        if calibrator_mode not in self.VALID_MODES:
            raise ValueError(
                f"calibrator_mode має бути одним з {self.VALID_MODES}, "
                f"отримано {calibrator_mode!r}"
            )

        # Базова фузійна мережа.
        self.base = AuthorTraceDetector(
            model_path=model_path, scaler_path=scaler_path,
            lm_name=lm_name, max_tokens=max_tokens,
            window_size=window_size, top_k=top_k, mcfg=mcfg,
        )

        self.calibrator_mode = calibrator_mode
        self.rules = RuleBasedCalibrator()

        # Логістичний калібратор — опціональний, потрібен лише в lr/hybrid.
        self.lr_calibrator = None
        if calibrator_mode in ("lr", "hybrid") and calibrator_path is not None:
            if os.path.exists(calibrator_path):
                self.lr_calibrator = Calibrator.load(calibrator_path)

    # ------------------------------------------------------------------

    @staticmethod
    def _extended_dict(text):
        """Витягує розширені стилометричні ознаки у вигляді словника."""
        full = extract_stylometric(text)
        extended = full[N_STYLOMETRIC_BASE:]
        return {
            name: float(val)
            for name, val in zip(EXTENDED_STYLOMETRIC_NAMES, extended)
        }

    def _calibrate(self, raw_p, ext_dict, source_lang, n_words):
        """Застосовує обраний калібратор. Повертає dict із calibrated_p
        та діагностикою."""
        mode = self.calibrator_mode

        if mode == "none":
            return {
                "calibrated_p": raw_p,
                "method": "none",
                "rules_fired": [],
            }

        # Завжди обчислюємо результат правил — він прозорий і безкоштовний.
        rule_result = rule_calibrate(
            raw_p, ext_dict, source_lang=source_lang, n_words=n_words,
        )
        rule_p = rule_result["calibrated_p"]

        if mode == "rules":
            return {
                "calibrated_p": rule_p,
                "method": "rules",
                "rules_fired": rule_result["rules_fired"],
                "rule_delta": rule_result["delta"],
                "llm_score": rule_result["llm_score"],
                "human_score": rule_result["human_score"],
            }

        # Для режимів lr / hybrid — підраховуємо LR-результат.
        lr_p = None
        if self.lr_calibrator is not None and self.lr_calibrator.is_trained:
            ext_values = np.array(
                [ext_dict[k] for k in EXTENDED_STYLOMETRIC_NAMES],
                dtype=np.float32,
            )
            x = Calibrator.build_input(
                raw_logit=_logit(raw_p),
                extended_features=ext_values,
                is_translated=(source_lang != "en"),
                is_short=(n_words < 80),
                is_ukrainian=(source_lang == "uk"),
            )
            lr_p = float(self.lr_calibrator.predict_proba(x))

        if mode == "lr":
            if lr_p is None:
                # Немає файлу LR — м'який fallback на правила.
                return {
                    "calibrated_p": rule_p,
                    "method": "rules_fallback",
                    "rules_fired": rule_result["rules_fired"],
                }
            return {
                "calibrated_p": lr_p,
                "method": "lr",
                "rules_fired": [],
            }

        # hybrid: якщо обидва калібратори згодні щодо класу — середнє,
        # інакше — rule-based (стійкіший до перенавчання).
        if lr_p is None:
            return {
                "calibrated_p": rule_p,
                "method": "rules_fallback",
                "rules_fired": rule_result["rules_fired"],
            }

        rule_label = rule_p >= 0.5
        lr_label = lr_p >= 0.5
        if rule_label == lr_label:
            return {
                "calibrated_p": (rule_p + lr_p) / 2.0,
                "method": "hybrid_agree",
                "rules_fired": rule_result["rules_fired"],
                "rule_p": rule_p,
                "lr_p": lr_p,
            }
        return {
            "calibrated_p": rule_p,
            "method": "hybrid_rules_win",
            "rules_fired": rule_result["rules_fired"],
            "rule_p": rule_p,
            "lr_p": lr_p,
        }

    # ------------------------------------------------------------------

    def predict_proba(self, text, auto_translate=True):
        """Повертає скориговану ймовірність штучного походження."""
        source_lang = detect_language(text)
        do_translate = auto_translate and source_lang != "en"

        raw_p = self.base.predict_proba(text, translate=do_translate)

        if self.calibrator_mode == "none":
            return raw_p

        ext_dict = self._extended_dict(text)
        n_words = len(text.split())
        result = self._calibrate(raw_p, ext_dict, source_lang, n_words)
        return result["calibrated_p"]

    def predict(self, text, threshold=0.5, auto_translate=True):
        p = self.predict_proba(text, auto_translate=auto_translate)
        return "AI" if p >= threshold else "Human"

    # ------------------------------------------------------------------

    def analyze(self, text, threshold=0.5, auto_translate=True,
                progress_callback=None):
        """Повний аналіз із калібруванням.

        Повертає словник з полями:

          * `verdict`               — "AI" або "Human";
          * `ai_probability`        — фінальна ймовірність штучного
                                      походження (після калібрування);
          * `confidence`            — впевненість моделі (|p - 0.5| × 2);
          * `raw_probability`       — ймовірність базової мережі;
          * `calibrated_probability` — ймовірність після калібратора;
          * `calibrator_used`       — bool, чи був застосований калібратор;
          * `calibrator_method`     — який метод спрацював (rules/lr/...);
          * `extended_features`     — dict з 10 розширеними стилометричними
                                      ознаками (sentence_length_cv та ін.);
          * `rules_fired`           — перелік спрацьованих правил;
          * `feature_contributions` — внески груп ознак (базова мережа);
          * `segments`              — посегментний аналіз;
          * `translation`           — інформація про мову та переклад.
        """
        source_lang = detect_language(text)
        translate = auto_translate and source_lang != "en"

        base_result = self.base.analyze(
            text, threshold=threshold, translate=translate,
            progress_callback=progress_callback,
        )

        raw_p = float(base_result["ai_probability"])
        ext_dict = self._extended_dict(text)
        n_words = len(text.split())

        base_result["raw_probability"] = raw_p
        base_result["extended_features"] = ext_dict

        if self.calibrator_mode == "none":
            base_result["calibrated_probability"] = raw_p
            base_result["calibrator_used"] = False
            base_result["calibrator_method"] = "none"
            base_result["rules_fired"] = []
            return base_result

        cal_result = self._calibrate(raw_p, ext_dict, source_lang, n_words)
        calibrated_p = cal_result["calibrated_p"]

        base_result["calibrated_probability"] = calibrated_p
        base_result["calibrator_used"] = True
        base_result["calibrator_method"] = cal_result["method"]
        base_result["rules_fired"] = cal_result.get("rules_fired", [])

        # Фінальні поля з каліброваним значенням.
        base_result["ai_probability"] = calibrated_p
        base_result["verdict"] = (
            "AI" if calibrated_p >= threshold else "Human"
        )
        base_result["confidence"] = abs(calibrated_p - 0.5) * 2.0

        # Рекалібрування сегментів. Базовий детектор класифікує кожен
        # сегмент сирим виходом фузійної мережі — без калібратора. Це
        # призводить до того, що сегменти з фрагментами Modern-LLM тексту
        # отримують raw_p близько 0 і відображаються як «людина», тоді як
        # загальний вердикт за калібратором — «штучний». Прогоняємо кожен
        # сегмент через rule-based калібратор зі стилометрією, обчисленою
        # на оригіналі сегмента, щоб посегментне виділення узгоджувалось
        # з фінальним вердиктом.
        segments = base_result.get("segments") or []
        if segments:
            seen_texts = set()
            recalibrated = []
            for seg in segments:
                seg_text = (seg.get("text") or "").strip()
                # Дедуплікація: однакові сегменти (часто виникають у текстах
                # із повторами, дублюючими заголовками чи службовими блоками)
                # подаються користувачу один раз.
                if not seg_text or seg_text in seen_texts:
                    continue
                seen_texts.add(seg_text)

                seg_raw_p = float(seg.get("ai_probability", 0.0))
                seg_words = len(seg_text.split())
                seg_lang = detect_language(seg_text)
                seg_ext = self._extended_dict(seg_text)
                seg_cal = self._calibrate(
                    seg_raw_p, seg_ext, seg_lang, seg_words,
                )
                seg["raw_ai_probability"] = seg_raw_p
                seg["ai_probability"] = seg_cal["calibrated_p"]
                recalibrated.append(seg)

            base_result["segments"] = recalibrated

            # Узгодження сегментів із загальним вердиктом. Сегменти
            # калібруються поодинці, але на коротких фрагментах стилометричні
            # сигнали слабші (правила вимагають мінімальної кількості слів).
            # Коли загальний вердикт — штучний текст, зсуваємо логіти
            # сегментів так, щоб їх середня дорівнювала фінальній каліброваній
            # ймовірності — це зберігає відносні відмінності між сегментами
            # (видно, які підозріліші), але масштаб шкали відповідає
            # загальній оцінці. Для людських текстів зсуву не робимо, щоб не
            # створювати хибних «гарячих» фрагментів.
            if recalibrated and calibrated_p >= threshold:
                import math
                def _logit(p):
                    p = min(max(p, 1e-6), 1 - 1e-6)
                    return math.log(p / (1 - p))
                def _sigmoid(z):
                    if z > 50: return 1.0
                    if z < -50: return 0.0
                    return 1.0 / (1.0 + math.exp(-z))
                target_logit = _logit(calibrated_p)
                cur_mean_logit = sum(
                    _logit(s["ai_probability"]) for s in recalibrated
                ) / len(recalibrated)
                shift = target_logit - cur_mean_logit
                for s in recalibrated:
                    s["ai_probability"] = _sigmoid(
                        _logit(s["ai_probability"]) + shift
                    )

        return base_result
