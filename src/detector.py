"""
Високорівнева обгортка детектора AuthorTrace.

Поєднує вилучення ознак, нормалізацію, нейромережу фузії та засоби
інтерпретації в єдиний інтерфейс для практичного використання.

Підтримка неангломовних текстів
-------------------------------
Модель навчена на англомовних текстах, тож для текстів іншими мовами
її поведінка не визначена. Щоб не зменшувати корисність системи для
україномовних користувачів, інтерфейс `analyze`/`predict_proba` підтримує
прозорий переклад через `src.translate.Translator`: при `translate=True`
неангломовний вхід автоматично перекладається англійською, і вже над
перекладом обчислюються ознаки. Користувач отримує як вердикт моделі,
так і інформацію про те, що саме було проаналізовано.
"""

import json

import numpy as np
import torch

from .features import FeatureExtractor
from .features.stylometric import N_STYLOMETRIC
from .features.perplexity import N_PERPLEXITY
from .model import FusionMLP, StandardScaler
from .explain import branch_contributions, segment_analysis


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
        """Створює перекладач один раз і кешує його."""
        if self._translator is None:
            from .translate import Translator
            self._translator = Translator(target="en")
        return self._translator

    def _maybe_translate(self, text, translate):
        """
        Повертає кортеж (текст_для_аналізу, інфо_про_переклад).
        Якщо переклад не виконувався — `інфо` має `translated=False`.
        """
        from .translate import detect_language

        source_lang = detect_language(text)
        info = {
            "source_language": source_lang,
            "translated": False,
            "translated_text": None,
        }
        if not translate or source_lang == "en":
            return text, info

        translator = self._get_translator()
        translated = translator.translate(text)
        info["translated"] = True
        info["translated_text"] = translated
        return translated, info

    # --- Базові операції ----------------------------------------------------

    def _scaled_vector(self, text):
        raw = self.extractor.extract(text)
        return self.scaler.transform(raw)

    @torch.no_grad()
    def predict_proba(self, text, translate=False):
        """
        Повертає ймовірність штучного походження тексту (0..1).

        :param translate: якщо True — неангломовний текст спершу буде
                          перекладено на англійську.
        """
        analysis_text, _ = self._maybe_translate(text, translate)
        x = self._scaled_vector(analysis_text)
        logit = self.model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        return float(torch.sigmoid(logit).item())

    def predict(self, text, threshold=0.5, translate=False):
        """Повертає мітку: 'AI' або 'Human'."""
        return ("AI"
                if self.predict_proba(text, translate=translate) >= threshold
                else "Human")

    # --- Розширений аналіз --------------------------------------------------

    def analyze(self, text, threshold=0.5, translate=False):
        """
        Повний аналіз тексту: вердикт, ймовірність, внески груп ознак,
        посегментна розбивка.

        :param translate: якщо True — для неангломовного тексту виконується
                          переклад на англійську, і весь подальший аналіз
                          (включно з посегментним) проводиться над
                          перекладом. У результаті повертається поле
                          `translation` з деталями.
        """
        analysis_text, trans_info = self._maybe_translate(text, translate)

        x = self._scaled_vector(analysis_text)
        logit = self.model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        prob = float(torch.sigmoid(logit).item())

        contrib = branch_contributions(self.model, x)
        # Посегментний аналіз — над тим самим текстом, що йшов у глобальну
        # модель, інакше сегменти й глобальний вердикт жили б у різних
        # мовних світах і не корелювали.
        segments = segment_analysis(self, analysis_text)

        return {
            "verdict": "AI" if prob >= threshold else "Human",
            "ai_probability": prob,
            "confidence": abs(prob - 0.5) * 2.0,
            "feature_contributions": contrib["shares"],
            "segments": segments,
            "translation": trans_info,
        }
