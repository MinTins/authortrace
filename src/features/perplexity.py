"""
Модуль вилучення перплексійних ознак на основі авторегресійної мовної моделі.

Ідея: текст, згенерований мовною моделлю, складається переважно з токенів,
які та сама або споріднена модель вважає високоймовірними. Тому згенерований
текст має нижчу перплексію та меншу «вибуховість» (burstiness) — розкид
несподіваності токенів. Окрім класичних метрик, обчислюється авторська
ознака — кривина перплексії за вікнами тексту.
"""

import numpy as np
import torch

# Назви ознак — потрібні для інтерпретації внеску у вердикт.
PERPLEXITY_NAMES = [
    "mean_log_prob",        # середня логарифмічна правдоподібність токенів
    "perplexity",           # перплексія всього тексту
    "burstiness",           # стандартне відхилення лог-імовірностей токенів
    "logprob_range",        # розмах лог-імовірностей (max - min)
    "window_ppl_std",       # кривина перплексії: розкид за вікнами (новизна)
    "top1_fraction",        # частка токенів, що збіглися з top-1 прогнозом
    "topk_fraction",        # частка токенів у top-k прогнозах моделі
    "mean_token_rank",      # середній ранг істинного токена в розподілі
]

N_PERPLEXITY = len(PERPLEXITY_NAMES)


class PerplexityExtractor:
    """Обгортка над мовною моделлю для обчислення перплексійних ознак."""

    def __init__(self, model, tokenizer, max_tokens=220, window_size=40, top_k=10):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.top_k = top_k

    @torch.no_grad()
    def _token_stats(self, input_ids):
        """
        Повертає масиви токен-рівневих лог-імовірностей та рангів істинних
        токенів, а також прихований стан останнього шару.
        """
        out = self.model(input_ids, output_hidden_states=True)
        logits = out.logits[0]                       # (seq, vocab)
        hidden = out.hidden_states[-1][0]            # (seq, dim)

        # Зсув: прогноз позиції t робиться за токенами до неї.
        log_probs = torch.log_softmax(logits[:-1], dim=-1)
        targets = input_ids[0][1:]

        true_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Ранг істинного токена в розподілі (0 — найімовірніший).
        ranks = (log_probs > true_lp.unsqueeze(1)).sum(dim=1)

        return true_lp.cpu().numpy(), ranks.cpu().numpy(), hidden

    def extract(self, text, return_hidden=False):
        """
        Обчислює вектор перплексійних ознак для тексту.

        :param return_hidden: якщо True — додатково повертає семантичний
                              ембединг (середнє за прихованим станом)
        """
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_tokens
        )
        input_ids = enc["input_ids"]

        if input_ids.shape[1] < 4:
            feats = np.zeros(N_PERPLEXITY, dtype=np.float32)
            if return_hidden:
                dim = self.model.config.hidden_size
                return feats, np.zeros(dim, dtype=np.float32)
            return feats

        true_lp, ranks, hidden = self._token_stats(input_ids)

        mean_lp = float(np.mean(true_lp))
        perplexity = float(np.exp(-mean_lp))
        burstiness = float(np.std(true_lp))
        lp_range = float(np.max(true_lp) - np.min(true_lp))

        # Авторська ознака — кривина перплексії за непересічними вікнами.
        window_ppls = []
        for i in range(0, len(true_lp), self.window_size):
            chunk = true_lp[i:i + self.window_size]
            if len(chunk) >= 5:
                window_ppls.append(float(np.exp(-np.mean(chunk))))
        window_ppl_std = float(np.std(window_ppls)) if len(window_ppls) > 1 else 0.0

        top1_fraction = float(np.mean(ranks == 0))
        topk_fraction = float(np.mean(ranks < self.top_k))
        mean_rank = float(np.mean(ranks))

        feats = np.array([
            mean_lp,
            min(perplexity, 1000.0),          # обмеження викидів
            burstiness,
            lp_range,
            min(window_ppl_std, 500.0),
            top1_fraction,
            topk_fraction,
            min(mean_rank, 5000.0),
        ], dtype=np.float32)

        if return_hidden:
            semantic = hidden.mean(dim=0).cpu().numpy().astype(np.float32)
            return feats, semantic
        return feats
