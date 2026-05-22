"""
Оркестратор вилучення ознак.

Поєднує три групи ознак у єдиний вектор фузії:
  * стилометричні  — інтерпретовані лінгвостатистичні характеристики;
  * перплексійні   — сигнали авторегресійної мовної моделі;
  * семантичні     — усереднений прихований стан того самого трансформера.

Одна мовна модель (distilgpt2) обслуговує і перплексійну, і семантичну
групи, що робить систему легкою та придатною до запуску на CPU.
"""

import os

# Гасимо інфо-повідомлення бекендів, які підтягуються транзитивно:
# TF (через keras в `deep-translator`) виводить попередження oneDNN та
# про застарілі API, а токенізатор HuggingFace попереджає при кожній
# токенізації послідовності, довшої за контекст моделі (це штатна
# ситуація — вище за стеком текст розбивається на фрагменти).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
import warnings

for _logger_name in ("tensorflow", "transformers", "absl"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
import transformers
transformers.logging.set_verbosity_error()
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from .stylometric import extract_stylometric, STYLOMETRIC_NAMES, N_STYLOMETRIC
from .perplexity import PerplexityExtractor, PERPLEXITY_NAMES, N_PERPLEXITY


class FeatureExtractor:
    """Вилучає повний вектор ознак для довільного тексту."""

    def __init__(self, lm_name="distilgpt2", max_tokens=220,
                 window_size=40, top_k=10):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(lm_name)
        self.model = GPT2LMHeadModel.from_pretrained(lm_name)
        self.model.eval()
        self.semantic_dim = self.model.config.hidden_size

        self.ppl = PerplexityExtractor(
            self.model, self.tokenizer,
            max_tokens=max_tokens, window_size=window_size, top_k=top_k,
        )

        # Межі підвекторів у спільному векторі фузії.
        self.bounds = {
            "stylometric": (0, N_STYLOMETRIC),
            "perplexity": (N_STYLOMETRIC, N_STYLOMETRIC + N_PERPLEXITY),
            "semantic": (N_STYLOMETRIC + N_PERPLEXITY,
                         N_STYLOMETRIC + N_PERPLEXITY + self.semantic_dim),
        }
        self.total_dim = N_STYLOMETRIC + N_PERPLEXITY + self.semantic_dim

    def feature_names(self):
        """Назви всіх ознак (семантичні — узагальнено)."""
        names = list(STYLOMETRIC_NAMES) + list(PERPLEXITY_NAMES)
        names += [f"sem_{i}" for i in range(self.semantic_dim)]
        return names

    def extract(self, text):
        """Повертає об'єднаний numpy-вектор ознак для одного тексту."""
        styl = extract_stylometric(text)
        ppl, semantic = self.ppl.extract(text, return_hidden=True)
        return np.concatenate([styl, ppl, semantic]).astype(np.float32)

    def extract_batch(self, texts, verbose=True):
        """Вилучає ознаки для списку текстів; повертає матрицю (N, total_dim)."""
        rows = []
        for i, t in enumerate(texts):
            rows.append(self.extract(t))
            if verbose and (i + 1) % 100 == 0:
                print(f"  вилучено ознак: {i + 1}/{len(texts)}")
        return np.vstack(rows)
