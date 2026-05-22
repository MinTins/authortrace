"""
Високорівнева обгортка детектора AuthorTrace.

Поєднує вилучення ознак, нормалізацію, нейромережу фузії та засоби
інтерпретації в єдиний інтерфейс для практичного використання.

Базова модель навчена на англомовному корпусі, тому для текстів іншими
мовами передбачено прозорий переклад через `src.translate.Translator`:
при `translate=True` неангломовний вхід автоматично перекладається
англійською. Сегменти при цьому формуються над текстом мовою оригіналу,
а класифікація — над перекладом кожного сегмента, тож виділення в UI
показується на тексті, з яким працює користувач.

Фінальний вердикт враховує як одиничний прогін моделі на повному тексті,
так і розподіл імовірностей по сегментах, що дозволяє коректно
класифікувати тексти зі змішаним або частково штучним змістом.
"""

import json

import numpy as np
import torch

from .features import FeatureExtractor
from .features.stylometric import N_STYLOMETRIC
from .features.perplexity import N_PERPLEXITY
from .model import FusionMLP, StandardScaler
from .explain import (
    branch_contributions,
    segment_analysis,
    segment_analysis_translated,
    aggregate_verdict,
)


class AuthorTraceDetector:
    """Детектор штучно згенерованих текстів."""

    def __init__(self, model_path, scaler_path, lm_name="distilgpt2",
                 max_tokens=220, window_size=40, top_k=10, mcfg=None):
        self.extractor = FeatureExtractor(
            lm_name=lm_name, max_tokens=max_tokens,
            window_size=window_size, top_k=top_k,
        )
        self.scaler = StandardScaler.from_dict(
            json.load(open(scaler_path, encoding="utf-8"))
        )

        dim_sem = self.extractor.semantic_dim
        mcfg = mcfg or {}
        self.model = FusionMLP(
            dim_styl=N_STYLOMETRIC, dim_ppl=N_PERPLEXITY, dim_sem=dim_sem,
            styl_hidden=mcfg.get("stylometric_hidden", 16),
            ppl_hidden=mcfg.get("perplexity_hidden", 16),
            sem_hidden=mcfg.get("semantic_hidden", 64),
            fusion_hidden=mcfg.get("fusion_hidden", 64),
            dropout=mcfg.get("dropout", 0.35),
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        # Ліниво створюваний перекладач (вмикається лише на запит).
        self._translator = None

    # --- Допоміжні методи перекладу ----------------------------------------

    def _get_translator(self):
        if self._translator is None:
            from .translate import Translator
            self._translator = Translator(target="en")
        return self._translator

    # --- Базові операції ----------------------------------------------------

    def _scaled_vector(self, text):
        raw = self.extractor.extract(text)
        return self.scaler.transform(raw)

    @torch.no_grad()
    def _global_prob(self, text):
        x = self._scaled_vector(text)
        logit = self.model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        return float(torch.sigmoid(logit).item()), x

    def predict_proba(self, text, translate=False):
        """
        Повертає ймовірність штучного походження тексту (0..1)
        за єдиним прогоном моделі (без посегментної агрегації).

        :param translate: якщо True — неангломовний текст спершу буде
                          перекладено англійською.
        """
        from .translate import detect_language
        if translate and detect_language(text) != "en":
            text = self._get_translator().translate(text)
        p, _ = self._global_prob(text)
        return p

    def predict(self, text, threshold=0.5, translate=False):
        """Повертає мітку: 'AI' або 'Human' за результатом `analyze`."""
        return self.analyze(text, threshold=threshold,
                            translate=translate)["verdict"]

    # --- Розширений аналіз --------------------------------------------------

    def analyze(self, text, threshold=0.5, translate=False,
                progress_callback=None):
        """
        Повний аналіз тексту: вердикт, ймовірність, внески груп ознак,
        посегментна розбивка та фінальна агрегація.

        :param translate: якщо True — для неангломовного тексту виконується
                          переклад на англійську. Сегменти при цьому
                          формуються над текстом мовою оригіналу, а
                          класифікація — над перекладом кожного сегмента.
        :param progress_callback: опційний callback `(stage, done, total)`
                                  для відображення прогресу під час
                                  довгих операцій (переклад тощо).
        """
        from .translate import detect_language

        source_lang = detect_language(text)
        do_translate = translate and source_lang != "en"

        trans_info = {
            "source_language": source_lang,
            "translated": False,
            "translated_text": None,
        }

        if do_translate:
            translator = self._get_translator()

            def _trans_progress(done, total):
                if progress_callback is not None:
                    progress_callback("translate", done, total)

            full_translated = translator.translate(
                text, progress_callback=_trans_progress
            )
            trans_info["translated"] = True
            trans_info["translated_text"] = full_translated
            analysis_text_for_global = full_translated
        else:
            translator = None
            analysis_text_for_global = text

        # Глобальний прогон моделі — даватиме одну з оцінок для агрегації.
        global_prob, x = self._global_prob(analysis_text_for_global)
        contrib = branch_contributions(self.model, x)

        # Посегментний аналіз: для перекладеного входу сегменти формуються
        # над оригіналом, але класифікуються через переклад.
        if do_translate:
            segments = segment_analysis_translated(self, text, translator)
        else:
            segments = segment_analysis(self, analysis_text_for_global)

        # Фінальна ймовірність — агрегація глобального результату й
        # посегментної розбивки.
        final_prob = aggregate_verdict(global_prob, segments, threshold)

        return {
            "verdict": "AI" if final_prob >= threshold else "Human",
            "ai_probability": final_prob,
            "global_ai_probability": global_prob,
            "confidence": abs(final_prob - 0.5) * 2.0,
            "feature_contributions": contrib["shares"],
            "segments": segments,
            "translation": trans_info,
        }
