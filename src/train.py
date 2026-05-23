"""
Модуль навчання гібридної нейромережі фузії ознак.

Реалізує цикл навчання з функцією втрат бінарної крос-ентропії,
оптимізатором Adam, регуляризацією через дропаут і L2-штраф, а також
ранньою зупинкою за значенням втрат на валідаційній вибірці.
"""

import numpy as np
import torch
import torch.nn as nn

from .features.stylometric import N_STYLOMETRIC_BASE as N_STYLOMETRIC
from .features.perplexity import N_PERPLEXITY
from .model import FusionMLP


def _batches(X, y, batch_size, shuffle, rng):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        sel = idx[i:i + batch_size]
        yield X[sel], y[sel]


def train_model(X_train, y_train, X_val, y_val, cfg, branch_mask=None):
    """
    Навчає FusionMLP та повертає найкращу модель і журнал навчання.

    :param branch_mask: необов'язковий словник {'stylometric','perplexity',
        'semantic'} -> bool для дослідження впливу окремих груп ознак
        (використовується в абляційному експерименті).
    """
    torch.manual_seed(cfg["seed"])
    rng = np.random.RandomState(cfg["seed"])

    dim_sem = X_train.shape[1] - N_STYLOMETRIC - N_PERPLEXITY
    mcfg, tcfg = cfg["model"], cfg["train"]

    # Маскування груп ознак для абляційного дослідження.
    if branch_mask is not None:
        X_train = _apply_mask(X_train, branch_mask, dim_sem)
        X_val = _apply_mask(X_val, branch_mask, dim_sem)

    model = FusionMLP(
        dim_styl=N_STYLOMETRIC, dim_ppl=N_PERPLEXITY, dim_sem=dim_sem,
        styl_hidden=mcfg["stylometric_hidden"],
        ppl_hidden=mcfg["perplexity_hidden"],
        sem_hidden=mcfg["semantic_hidden"],
        fusion_hidden=mcfg["fusion_hidden"],
        dropout=mcfg["dropout"],
    )

    opt = torch.optim.Adam(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(tcfg["epochs"]):
        model.train()
        ep_loss = 0.0
        n_batch = 0
        for xb, yb in _batches(Xtr.numpy(), ytr.numpy(),
                               tcfg["batch_size"], True, rng):
            xb = torch.tensor(xb, dtype=torch.float32)
            yb = torch.tensor(yb, dtype=torch.float32)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batch += 1

        # Оцінювання на валідаційній вибірці.
        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = loss_fn(val_logits, yva).item()
            val_pred = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = (val_pred == yva).float().mean().item()

        history["train_loss"].append(ep_loss / max(1, n_batch))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= tcfg["early_stopping_patience"]:
                print(f"  рання зупинка на епосі {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  епоха {epoch + 1:3d} | "
                  f"train_loss={history['train_loss'][-1]:.4f} | "
                  f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _apply_mask(X, mask, dim_sem):
    """Обнуляє групи ознак, вимкнені в абляційному дослідженні."""
    X = X.copy()
    a, b = N_STYLOMETRIC, N_STYLOMETRIC + N_PERPLEXITY
    if not mask.get("stylometric", True):
        X[:, :a] = 0.0
    if not mask.get("perplexity", True):
        X[:, a:b] = 0.0
    if not mask.get("semantic", True):
        X[:, b:] = 0.0
    return X
