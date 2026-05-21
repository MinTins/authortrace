"""
Модуль оцінювання якості та візуалізації результатів.

Обчислює стандартні метрики класифікації та будує чорно-білі графіки,
придатні для друку в тексті курсової роботи.
"""

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

# Єдиний чорно-білий стиль для всіх ілюстрацій.
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.color": "0.8",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "0.2",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


@torch.no_grad()
def predict_scores(model, X):
    """Повертає ймовірності класу 'штучний текст' для матриці ознак."""
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def compute_metrics(y_true, y_score, threshold=0.5):
    """Обчислює набір метрик якості бінарної класифікації."""
    y_true = np.asarray(y_true)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# --- Графіки ---------------------------------------------------------------

def plot_training(history, path):
    """Криві навчання: втрати на тренувальній та валідаційній вибірках."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ep = range(1, len(history["train_loss"]) + 1)
    ax.plot(ep, history["train_loss"], color="black",
            linewidth=1.6, label="Навчальна вибірка")
    ax.plot(ep, history["val_loss"], color="black", linestyle="--",
            linewidth=1.6, label="Валідаційна вибірка")
    ax.set_xlabel("Епоха навчання")
    ax.set_ylabel("Втрати (бінарна крос-ентропія)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(y_true, y_score, path):
    """ROC-крива з позначенням площі під нею."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot(fpr, tpr, color="black", linewidth=1.8,
            label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Частка хибнопозитивних (FPR)")
    ax.set_ylabel("Частка істиннопозитивних (TPR)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, path):
    """Матриця плутанини у відтінках сірого."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max())
    labels = ["Людина", "Штучний"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Прогноз моделі")
    ax.set_ylabel("Справжній клас")
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=14)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ablation(ablation, path):
    """Стовпчикова діаграма точності для різних наборів груп ознак."""
    names = list(ablation.keys())
    accs = [ablation[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bars = ax.bar(range(len(names)), accs, color="0.35",
                  edgecolor="black", width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Точність (Accuracy)")
    ax.set_ylim(0.5, 1.0)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.005,
                f"{a:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_feature_distribution(feat_human, feat_ai, name, path):
    """Гістограми розподілу однієї ознаки для двох класів."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bins = 30
    ax.hist(feat_human, bins=bins, color="0.25", alpha=0.7,
            label="Людина", edgecolor="black", linewidth=0.4)
    ax.hist(feat_ai, bins=bins, color="0.7", alpha=0.7,
            label="Штучний текст", edgecolor="black", linewidth=0.4)
    ax.set_xlabel(name)
    ax.set_ylabel("Кількість текстів")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
