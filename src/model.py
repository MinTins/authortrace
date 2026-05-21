"""
Гібридна нейромережа фузії ознак.

Архітектура реалізує проміжну (intermediate) фузію: кожна група ознак
спершу обробляється власним підкодувальником, після чого приховані
представлення об'єднуються та подаються у спільний класифікаційний блок.
Такий підхід дозволяє моделі окремо «зважувати» внесок стилометричних,
перплексійних і семантичних сигналів.
"""

import numpy as np
import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Багатогілкова нейромережа класифікації текстів за походженням."""

    def __init__(self, dim_styl, dim_ppl, dim_sem,
                 styl_hidden=16, ppl_hidden=16, sem_hidden=64,
                 fusion_hidden=64, dropout=0.35):
        super().__init__()
        self.dim_styl = dim_styl
        self.dim_ppl = dim_ppl
        self.dim_sem = dim_sem

        # Підкодувальники окремих груп ознак.
        self.enc_styl = nn.Sequential(
            nn.Linear(dim_styl, styl_hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.enc_ppl = nn.Sequential(
            nn.Linear(dim_ppl, ppl_hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.enc_sem = nn.Sequential(
            nn.Linear(dim_sem, sem_hidden), nn.ReLU(), nn.Dropout(dropout)
        )

        # Спільний блок фузії та класифікації.
        fused = styl_hidden + ppl_hidden + sem_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fused, fusion_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def _split(self, x):
        a = self.dim_styl
        b = a + self.dim_ppl
        return x[:, :a], x[:, a:b], x[:, b:]

    def branch_features(self, x):
        """Повертає приховані представлення кожної з трьох гілок."""
        xs, xp, xm = self._split(x)
        return self.enc_styl(xs), self.enc_ppl(xp), self.enc_sem(xm)

    def forward(self, x):
        hs, hp, hm = self.branch_features(x)
        fused = torch.cat([hs, hp, hm], dim=1)
        return self.classifier(fused).squeeze(-1)   # логіт


class StandardScaler:
    """Z-нормалізація ознак; зберігається разом з моделлю."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.mean = np.array(d["mean"], dtype=np.float32)
        s.std = np.array(d["std"], dtype=np.float32)
        return s
