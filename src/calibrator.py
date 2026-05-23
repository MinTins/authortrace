"""
Калібратор пост-обробки — логістична регресія.

Калібратор приймає на вхід ймовірність базової фузійної мережі та
розширені стилометричні ознаки і повертає скориговану ймовірність
штучного походження. Він використовується як один з режимів пост-
обробки в детекторі (`calibrator_mode='lr'` або `'hybrid'`); за
замовчуванням активний rule-based калібратор (див. `calibrator_rules.py`).

Логістична регресія обрана навмисно — вона інтерпретована, не схильна
до перенавчання на маленькому калібрувальному наборі, і її коефіцієнти
дають чіткий звіт про внески ознак у фінальне рішення.

Вхідний вектор калібратора (14 елементів):
  • raw_logit — логіт сирої ймовірності базової моделі;
  • 10 розширених стилометричних ознак (sentence_length_cv ... formal_register);
  • is_translated, is_short, is_ukrainian — контекстні прапорці.
"""

import json
import os

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


class Calibrator:
    """Логістичний калібратор для пост-обробки вердикту базової моделі."""

    # Назви вхідних ознак калібратора. Порядок критичний для збереження
    # та завантаження вагів.
    INPUT_NAMES = [
        # Сирий вихід базової моделі.
        "raw_logit",
        # Розширені стилометричні ознаки (10).
        "sentence_length_cv",
        "paragraph_balance",
        "connector_entropy",
        "hedge_density",
        "nominalization_ratio",
        "parallel_structure",
        "em_dash_ratio",
        "comma_run_score",
        "lexical_uniformity",
        "formal_register",
        # Контекстні прапорці.
        "is_translated",
        "is_short",
        "is_ukrainian",
    ]

    DIM = len(INPUT_NAMES)

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.scaler_mean = None
        self.scaler_std = None
        self.l2 = 1.0
        self.is_trained = False

    # --- Збирання вектора ознак --------------------------------------------

    @staticmethod
    def build_input(raw_logit, extended_features, *,
                    is_translated=False, is_short=False, is_ukrainian=False):
        """Збирає вхідний вектор для калібратора.

        :param raw_logit: логіт базової моделі (logit(raw_p))
        :param extended_features: dict або np.ndarray з 10 розширеними
            стилометричними ознаками у порядку EXTENDED_STYLOMETRIC_NAMES
        :param is_translated/is_short/is_ukrainian: bool прапорці
        :return: np.ndarray розмірності DIM
        """
        if isinstance(extended_features, dict):
            ext_arr = np.array([
                extended_features["sentence_length_cv"],
                extended_features["paragraph_balance"],
                extended_features["connector_entropy"],
                extended_features["hedge_density"],
                extended_features["nominalization_ratio"],
                extended_features["parallel_structure"],
                extended_features["em_dash_ratio"],
                extended_features["comma_run_score"],
                extended_features["lexical_uniformity"],
                extended_features["formal_register"],
            ], dtype=np.float32)
        else:
            ext_arr = np.asarray(extended_features, dtype=np.float32)

        context = np.array([
            1.0 if is_translated else 0.0,
            1.0 if is_short else 0.0,
            1.0 if is_ukrainian else 0.0,
        ], dtype=np.float32)

        return np.concatenate(
            [[raw_logit], ext_arr, context]
        ).astype(np.float32)

    # --- Нормалізація ------------------------------------------------------

    def _scale(self, X):
        if self.scaler_mean is None:
            return X
        return (X - self.scaler_mean) / (self.scaler_std + 1e-8)

    # --- Прогнозування -----------------------------------------------------

    def predict_proba(self, x):
        """Повертає скориговану ймовірність штучного походження.

        :param x: np.ndarray розмірності DIM
        :return: float у [0, 1]
        """
        if not self.is_trained:
            return float(_sigmoid(x[0]))
        x_scaled = self._scale(x)
        z = float(np.dot(self.weights, x_scaled) + self.bias)
        return float(_sigmoid(z))

    # --- Навчання ----------------------------------------------------------

    def fit(self, X, y, l2=1.0, lr=0.5, n_iter=2000, verbose=False):
        """Навчає логістичну регресію на калібрувальному наборі.

        Реалізована ванільна градієнтна оптимізація — на десятках
        прикладів і ~14 ознаках це майже миттєво й не вимагає sklearn.

        :param X: np.ndarray (N, DIM)
        :param y: np.ndarray (N,) — мітки {0, 1}
        :param l2: коефіцієнт L2-регуляризації
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, dim = X.shape
        assert dim == self.DIM, f"очікувано {self.DIM} ознак, отримано {dim}"

        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0) + 1e-8
        Xs = (X - self.scaler_mean) / self.scaler_std

        w = np.zeros(dim, dtype=np.float64)
        b = 0.0

        for it in range(n_iter):
            z = Xs @ w + b
            p = _sigmoid(z)
            err = p - y

            grad_w = (Xs.T @ err) / n + l2 * w / n
            grad_b = float(err.mean())

            w -= lr * grad_w
            b -= lr * grad_b

            if verbose and (it + 1) % 500 == 0:
                # Стабілізована BCE, щоб не отримувати -inf при p біля 0/1.
                bce = float(np.mean(
                    np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0) - z * y
                ))
                acc = float(((p >= 0.5).astype(int) == y).mean())
                print(f"  iter {it + 1:5d}  loss={bce:.4f}  acc={acc:.4f}")

        self.weights = w.astype(np.float32)
        self.bias = float(b)
        self.l2 = l2
        self.is_trained = True
        return self

    # --- Серіалізація ------------------------------------------------------

    def to_dict(self):
        return {
            "weights": None if self.weights is None else self.weights.tolist(),
            "bias": self.bias,
            "scaler_mean": (
                None if self.scaler_mean is None else self.scaler_mean.tolist()
            ),
            "scaler_std": (
                None if self.scaler_std is None else self.scaler_std.tolist()
            ),
            "l2": self.l2,
            "is_trained": self.is_trained,
            "input_names": self.INPUT_NAMES,
        }

    @classmethod
    def from_dict(cls, d):
        c = cls()
        c.weights = (
            None if d["weights"] is None
            else np.array(d["weights"], dtype=np.float32)
        )
        c.bias = float(d["bias"])
        c.scaler_mean = (
            None if d["scaler_mean"] is None
            else np.array(d["scaler_mean"], dtype=np.float64)
        )
        c.scaler_std = (
            None if d["scaler_std"] is None
            else np.array(d["scaler_std"], dtype=np.float64)
        )
        c.l2 = float(d.get("l2", 1.0))
        c.is_trained = bool(d.get("is_trained", False))
        return c

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"калібратор не знайдено: {path}")
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # --- Інтерпретація -----------------------------------------------------

    def coefficient_report(self):
        """Звіт про коефіцієнти калібратора (для діагностики)."""
        if not self.is_trained:
            return "Калібратор не навчений."
        lines = [
            f"Калібратор (логістична регресія, L2={self.l2})",
            f"Bias: {self.bias:+.4f}",
            "",
            f"{'Ознака':<28} {'Вага':>10}",
            "-" * 40,
        ]
        for name, w in zip(self.INPUT_NAMES, self.weights):
            lines.append(f"{name:<28} {float(w):>+10.4f}")
        return "\n".join(lines)
