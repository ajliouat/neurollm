"""Baseline models for motor imagery classification.

Implements:
  • CSP + SVM (MNE-Python + scikit-learn)
  • EEGNet (~2.6 K params, Lawhern 2018)
  • Vanilla Transformer (same arch as NeuRoLLM, no pre-training)
"""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.transformer import NeuRoLLM


# =====================================================================
#  CSP + SVM
# =====================================================================

class CSPFilter:
    """Common Spatial Patterns (CSP) implementation.

    Simplified 2-class extension to multi-class via one-vs-rest.
    For M classes uses M*(M-1)/2 pairs, k filters per pair → features.
    """

    def __init__(self, n_components: int = 6) -> None:
        self.n_components = n_components
        self.filters_: list[NDArray] | None = None

    def _csp_pair(
        self, X1: NDArray, X2: NDArray
    ) -> NDArray:
        """Compute CSP filters for two classes."""
        C1 = np.mean([x @ x.T for x in X1], axis=0)
        C2 = np.mean([x @ x.T for x in X2], axis=0)
        C1 /= np.trace(C1)
        C2 /= np.trace(C2)

        Cc = C1 + C2
        eigvals, eigvecs = np.linalg.eigh(Cc)
        # Whitening
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        W = eigvecs / np.sqrt(eigvals + 1e-10)

        S1 = W.T @ C1 @ W
        eigvals_s, eigvecs_s = np.linalg.eigh(S1)
        idx = np.argsort(eigvals_s)[::-1]
        eigvecs_s = eigvecs_s[:, idx]

        filters = (eigvecs_s.T @ W.T)  # (C, C)
        # Take first and last n_components/2
        k = self.n_components // 2
        selected = np.vstack([filters[:k], filters[-k:]])
        return selected

    def fit(self, X: NDArray, y: NDArray) -> "CSPFilter":
        """Fit CSP filters.

        Parameters
        ----------
        X : array (N, C, T)
        y : array (N,)
        """
        classes = np.unique(y)
        self.filters_ = []
        for i, c1 in enumerate(classes):
            for c2 in classes[i + 1 :]:
                X1 = X[y == c1]
                X2 = X[y == c2]
                f = self._csp_pair(X1, X2)
                self.filters_.append(f)
        self.filters_ = np.vstack(self.filters_)
        return self

    def transform(self, X: NDArray) -> NDArray:
        """Apply CSP spatial filters and extract log-variance features.

        Parameters
        ----------
        X : array (N, C, T)

        Returns
        -------
        features : array (N, n_features)
        """
        if self.filters_ is None:
            raise RuntimeError("CSPFilter not fitted")
        projected = np.array([self.filters_ @ x for x in X])  # (N, F, T)
        var = np.var(projected, axis=-1)  # (N, F)
        return np.log(var + 1e-10)


def train_csp_svm(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    n_components: int = 6,
) -> Dict[str, Any]:
    """Train and evaluate CSP + SVM baseline.

    Parameters
    ----------
    X_train, X_test : (N, C, T) float arrays
    y_train, y_test : (N,) int arrays
    n_components : int — CSP components

    Returns
    -------
    dict with accuracy, predictions, labels
    """
    csp = CSPFilter(n_components=n_components)
    csp.fit(X_train, y_train)

    feat_train = csp.transform(X_train)
    feat_test = csp.transform(X_test)

    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test = scaler.transform(feat_test)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(feat_train, y_train)

    preds = svm.predict(feat_test)
    acc = float(np.mean(preds == y_test))

    return {
        "accuracy": acc,
        "predictions": preds,
        "labels": y_test,
        "model_name": "CSP+SVM",
    }


# =====================================================================
#  EEGNet (~2.6K params, Lawhern 2018)
# =====================================================================

class EEGNet(nn.Module):
    """EEGNet — compact CNN for EEG classification.

    Simplified version of Lawhern et al. 2018.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_samples : int
        Temporal length.
    n_classes : int
        Number of output classes.
    F1 : int
        Number of temporal filters.
    D : int
        Depth multiplier for depthwise conv.
    F2 : int
        Number of separable filters.
    dropout : float
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 875,
        n_classes: int = 4,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        # Block 1: Temporal + Spatial
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable conv
        self.separable = nn.Conv2d(
            F1 * D, F2, (1, 16), padding=(0, 8), bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self._features(dummy)
            flat_size = dummy.shape[1]

        self.classifier = nn.Linear(flat_size, n_classes)

    def _features(self, x: Tensor) -> Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, C, T) — raw multi-channel EEG

        Returns
        -------
        logits : Tensor (B, n_classes)
        """
        x = x.unsqueeze(1)  # (B, 1, C, T)
        features = self._features(x)
        return self.classifier(features)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_eegnet(
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_channels: int = 22,
    n_samples: int = 875,
    n_classes: int = 4,
    n_epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Train and evaluate EEGNet baseline."""
    import copy

    model = EEGNet(n_channels, n_samples, n_classes, dropout=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx).argmax(1)
                correct += (preds == by).sum().item()
                total += by.shape[0]
        acc = correct / max(1, total)

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "accuracy": best_acc,
        "model_name": "EEGNet",
        "num_parameters": model.num_parameters,
    }


# =====================================================================
#  Vanilla Transformer (no pre-training)
# =====================================================================

def train_vanilla_transformer(
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_channels: int = 22,
    patch_size: int = 50,
    d_model: int = 256,
    n_heads: int = 4,
    d_ff: int = 512,
    n_layers: int = 6,
    n_classes: int = 4,
    n_epochs: int = 100,
    lr: float = 5e-5,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Train a vanilla transformer (same arch as NeuRoLLM, random init)."""
    import copy

    model = NeuRoLLM(
        n_channels=n_channels,
        patch_size=patch_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        n_classes=n_classes,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx, mode="finetune")
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx, mode="finetune").argmax(1)
                correct += (preds == by).sum().item()
                total += by.shape[0]
        acc = correct / max(1, total)

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "accuracy": best_acc,
        "model_name": "VanillaTransformer",
        "num_parameters": model.num_parameters,
    }
