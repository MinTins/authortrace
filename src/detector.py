"""
Високорівнева обгортка детектора AuthorTrace.

Поєднує вилучення ознак, нормалізацію, нейромережу фузії та засоби
інтерпретації в єдиний інтерфейс для практичного використання.
"""

import json
import os

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

    # --- Базові операції ----------------------------------------------------

    def _scaled_vector(self, text):
        raw = self.extractor.extract(text)
        return self.scaler.transform(raw)

    @torch.no_grad()
    def predict_proba(self, text):
        """Повертає ймовірність штучного походження тексту (0..1)."""
        x = self._scaled_vector(text)
        logit = self.model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        return float(torch.sigmoid(logit).item())

    def predict(self, text, threshold=0.5):
        """Повертає мітку: 'AI' або 'Human'."""
        return "AI" if self.predict_proba(text) >= threshold else "Human"

    # --- Розширений аналіз --------------------------------------------------

    def analyze(self, text, threshold=0.5):
        """
        Повний аналіз тексту: вердикт, ймовірність, внески груп ознак
        та посегментна розбивка.
        """
        x = self._scaled_vector(text)
        logit = self.model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        prob = float(torch.sigmoid(logit).item())

        contrib = branch_contributions(self.model, x)
        segments = segment_analysis(self, text)

        return {
            "verdict": "AI" if prob >= threshold else "Human",
            "ai_probability": prob,
            "confidence": abs(prob - 0.5) * 2.0,
            "feature_contributions": contrib["shares"],
            "segments": segments,
        }
