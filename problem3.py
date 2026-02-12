"""
HW3 - Problem 3: Linear SVM via hinge loss minimization (from scratch)

Objective:
  J(w) = 0.5 * ||w[1:]||^2 + C * (1/N) * sum max(0, 1 - y_i*(w0 + x_i @ w[1:]))

y in {-1, +1}. Bias w0 is not regularized.

Allowed: numpy (required), optionally scipy/pandas for loading.
Do NOT use sklearn models for training.

Autograder imports these functions directly. Keep names/signatures unchanged.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np


def hinge_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, C: float = 1.0) -> float:
    """
    Compute the hinge-loss SVM objective.

    Parameters
    ----------
    X : (N, d)
    y : (N,) in {-1, +1}
    w : (d+1,)
    C : float > 0

    Returns
    -------
    loss : float
    """
    # TODO
    weights = w[1:]
    bias = w[0]
    output = bias + X @ weights
    loss = np.sum(np.maximum(0, 1-y*output))
    reg_loss = 1 / 2 * np.sum(weights**2)
    return float(C * loss + reg_loss)


def hinge_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, C: float = 1.0) -> np.ndarray:
    """
    Subgradient of hinge_loss w.r.t. w.

    At margin exactly 1, any subgradient is acceptable; choose a deterministic convention.

    Returns
    -------
    grad : np.ndarray, shape (d+1,)
    """
    # TODO
    weights = w[1:]
    bias = w[0]
    out = bias + X @ weights
    active_mask = (1-y*out > 0)
    grad_weights = -X.T @ (y * active_mask) + weights
    grad_bias = -np.sum(y * active_mask) * C
    return np.concatenate([grad_bias, grad_weights])


def predict_svm(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Predict labels in {-1, +1} using sign(w0 + X @ w[1:]).

    Break ties (score == 0) as +1.

    Returns
    -------
    yhat : np.ndarray, shape (N,)
        Entries in {-1, +1}.
    """
    # TODO
    raise NotImplementedError


def train_svm_hinge(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    step_size: float = 0.1,
    max_epochs: int = 5000,
    tol: float = 1e-6,
    batch_size: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Train hinge-loss SVM starting from w = zeros(d+1).

    If batch_size == 0: full-batch GD per epoch.
    Else: mini-batch GD with deterministic shuffling controlled by seed.

    Early stop if abs(obj_t - obj_{t-1}) < tol (objective computed on full data).

    Returns dict with at least:
      - "w": np.ndarray (d+1,)
      - "loss_history": list[float] objective value per epoch
      - "err_history": list[float] training error rate
      - "epochs": int
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # Quick self-check (not graded): binary iris (0 vs 1) if present
    try:
        data = np.loadtxt("data/iris.txt")
        Xall = data[:, :4]
        yall = data[:, 4].astype(int)

        mask = (yall == 0) | (yall == 1)
        X = Xall[mask]
        y01 = yall[mask]
        y = np.where(y01 == 0, -1, +1).astype(int)

        # Standardize using subset
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

        out = train_svm_hinge(X, y, C=1.0, step_size=0.05, max_epochs=5000, tol=1e-9, batch_size=0, seed=0)
        w = out["w"]
        yhat = predict_svm(X, w)
        print("Train acc:", float(np.mean(yhat == y)))
        print("Final objective:", out["loss_history"][-1], "epochs:", out["epochs"])
    except FileNotFoundError:
        print("data/iris.txt not found. Put dataset in data/ and rerun.")
    except NotImplementedError:
        print("Implement the TODOs first.")
