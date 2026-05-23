"""
Оркестратор вилучення ознак.

Поєднує три групи ознак у єдиний вектор фузії:
  * стилометричні  — інтерпретовані лінгвостатистичні характеристики;
  * перплексійні   — сигнали авторегресійної мовної моделі;
  * семантичні     — усереднений прихований стан того самого трансформера.

Одна мовна модель (distilgpt2) обслуговує і перплексійну, і семантичну
групи, що робить систему легкою та придатною до запуску на CPU.

Зверніть увагу: `extract` повертає вектор фіксованої розмірності, який
подається у фузійну нейромережу. Розширений набір стилометричних ознак
(використовується калібратором пост-обробки) вилучається окремим
методом `extract_extended_stylometric`, що не змінює формат вхідного
вектора нейромережі.
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

from .stylometric import (
    extract_stylometric,
    STYLOMETRIC_NAMES,
    EXTENDED_STYLOMETRIC_NAMES,
    N_STYLOMETRIC,
    N_STYLOMETRIC_BASE,
    N_STYLOMETRIC_EXTENDED,
)
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

        # Межі підвекторів у спільному векторі фузії. Фузійна нейромережа
        # приймає лише БАЗОВУ підгрупу стилометричних ознак; розширений
        # набір використовується калібратором пост-обробки окремо.
        self.bounds = {
            "stylometric": (0, N_STYLOMETRIC_BASE),
            "perplexity": (
                N_STYLOMETRIC_BASE,
                N_STYLOMETRIC_BASE + N_PERPLEXITY,
            ),
            "semantic": (
                N_STYLOMETRIC_BASE + N_PERPLEXITY,
                N_STYLOMETRIC_BASE + N_PERPLEXITY + self.semantic_dim,
            ),
        }
        self.total_dim = (
            N_STYLOMETRIC_BASE + N_PERPLEXITY + self.semantic_dim
        )

    def feature_names(self):
        """Назви ознак, що подаються до фузійної мережі."""
        base_names = STYLOMETRIC_NAMES[:N_STYLOMETRIC_BASE]
        names = list(base_names) + list(PERPLEXITY_NAMES)
        names += [f"sem_{i}" for i in range(self.semantic_dim)]
        return names

    def extract(self, text):
        """Об'єднаний вектор ознак для одного тексту (вхід фузійної мережі).

        Повертає `total_dim` значень: базова стилометрія (15) + перплексія
        (8) + семантика (~768). Розширені стилометричні ознаки (15..24)
        тут НЕ включаються — вони вилучаються окремо для калібратора.
        """
        styl_full = extract_stylometric(text)
        styl_base = styl_full[:N_STYLOMETRIC_BASE]
        ppl, semantic = self.ppl.extract(text, return_hidden=True)
        return np.concatenate(
            [styl_base, ppl, semantic]
        ).astype(np.float32)

    def extract_extended_stylometric(self, text):
        """Розширений стилометричний підвектор (вхід калібратора).

        Повертає `N_STYLOMETRIC_EXTENDED` значень у порядку
        `EXTENDED_STYLOMETRIC_NAMES`. Ці ознаки обчислюються над
        ОРИГІНАЛОМ тексту, без машинного перекладу, що зберігає
        авторську стилістику.
        """
        styl_full = extract_stylometric(text)
        return styl_full[N_STYLOMETRIC_BASE:].astype(np.float32)

    def extract_batch(self, texts, verbose=True):
        """Вилучає ознаки для списку текстів; повертає матрицю (N, total_dim)."""
        rows = []
        for i, t in enumerate(texts):
            rows.append(self.extract(t))
            if verbose and (i + 1) % 100 == 0:
                print(f"  вилучено ознак: {i + 1}/{len(texts)}")
        return np.vstack(rows)
