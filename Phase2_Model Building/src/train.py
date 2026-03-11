from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ALL_MODEL_NAMES = [
    "baseline_persistence",
    "logistic_regression",
    "random_forest",
    "lstm",
]

MODEL_ALIASES = {
    "baseline": "baseline_persistence",
    "baseline_persistence": "baseline_persistence",
    "logistic": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "rf": "random_forest",
    "random_forest": "random_forest",
    "lstm": "lstm",
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_model_selection(models: Optional[Sequence[str]]) -> List[str]:
    if not models:
        return ALL_MODEL_NAMES.copy()

    selected: List[str] = []
    for raw in models:
        token = str(raw).strip().lower()
        if token == "all":
            return ALL_MODEL_NAMES.copy()
        if token not in MODEL_ALIASES:
            raise ValueError(f"Unknown model '{raw}'. Valid: all, baseline, logistic, rf, lstm")
        name = MODEL_ALIASES[token]
        if name not in selected:
            selected.append(name)

    return selected


class TabularPreprocessor:
    def __init__(self, numeric_features: Sequence[str], categorical_features: Sequence[str]):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)

        self.numeric_median: Dict[str, float] = {}
        self.numeric_mean: Dict[str, float] = {}
        self.numeric_std: Dict[str, float] = {}
        self.categorical_values: Dict[str, List[str]] = {}
        self.output_feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        self.output_feature_names = []

        for col in self.numeric_features:
            s = pd.to_numeric(df[col], errors="coerce")
            median = float(s.median()) if s.notna().any() else 0.0
            s = s.fillna(median)
            mean = float(s.mean())
            std = float(s.std(ddof=0))
            if std == 0.0 or np.isnan(std):
                std = 1.0

            self.numeric_median[col] = median
            self.numeric_mean[col] = mean
            self.numeric_std[col] = std
            self.output_feature_names.append(col)

        for col in self.categorical_features:
            s = df[col].fillna("__MISSING__").astype(str).str.strip()
            vals = sorted(s.unique().tolist())
            if not vals:
                vals = ["__MISSING__"]
            self.categorical_values[col] = vals
            for val in vals:
                self.output_feature_names.append(f"{col}__{val}")

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        blocks: List[np.ndarray] = []

        for col in self.numeric_features:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.fillna(self.numeric_median[col])
            arr = ((s.to_numpy(dtype=np.float32) - self.numeric_mean[col]) / self.numeric_std[col]).reshape(n, 1)
            blocks.append(arr.astype(np.float32))

        for col in self.categorical_features:
            vals = self.categorical_values[col]
            s = df[col].fillna("__MISSING__").astype(str).str.strip()
            cat = pd.Categorical(s, categories=vals)
            codes = cat.codes
            onehot = np.zeros((n, len(vals)), dtype=np.float32)
            valid_mask = codes >= 0
            rows = np.arange(n)[valid_mask]
            onehot[rows, codes[valid_mask]] = 1.0
            blocks.append(onehot)

        if not blocks:
            return np.zeros((n, 0), dtype=np.float32)
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def to_dict(self) -> Dict[str, object]:
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "numeric_median": self.numeric_median,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "categorical_values": self.categorical_values,
            "output_feature_names": self.output_feature_names,
        }


def build_lstm_sequences(
    df: pd.DataFrame,
    X_tab: np.ndarray,
    sequence_length: int,
    include_target: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sequence_length <= 1:
        raise ValueError("LSTM sequence_length must be > 1")

    base = df.reset_index(drop=True).copy()
    base["_row_id"] = np.arange(len(base))
    base["_time"] = pd.to_datetime(base["time_id"], errors="coerce")
    group_keys = ["region_id"]
    if "crime_group" in base.columns:
        group_keys.append("crime_group")
    base = base.sort_values(group_keys + ["_time"]).reset_index(drop=True)

    y_all = df.reset_index(drop=True)["y"].astype(int).to_numpy()

    seq_list: List[np.ndarray] = []
    y_list: List[int] = []
    row_id_list: List[int] = []

    feature_dim = int(X_tab.shape[1])

    for _, group in base.groupby(group_keys, sort=False):
        rows = group["_row_id"].to_numpy(dtype=int)
        if len(rows) == 0:
            continue

        for end in range(len(rows)):
            start = max(0, end - sequence_length + 1)
            window = rows[start : end + 1]
            seq = X_tab[window]
            if len(window) < sequence_length:
                pad_len = sequence_length - len(window)
                # Left-pad with zeros so every row has one sequence and model outputs stay aligned.
                pad = np.zeros((pad_len, feature_dim), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)

            target_row = rows[end]
            seq_list.append(seq)
            row_id_list.append(target_row)
            if include_target:
                y_list.append(int(y_all[target_row]))

    if not seq_list:
        empty_seq = np.zeros((0, sequence_length, X_tab.shape[1]), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.float32)
        empty_idx = np.zeros((0,), dtype=np.int64)
        return empty_seq, empty_y, empty_idx

    seq_x = np.stack(seq_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32) if include_target else np.zeros((len(seq_list),), dtype=np.float32)
    row_idx = np.asarray(row_id_list, dtype=np.int64)
    return seq_x, y, row_idx


class TorchLogisticRegressionModel:
    def __init__(self, input_dim: int, seed: int, lr: float = 1e-2, epochs: int = 220, batch_size: int = 256):
        self.input_dim = input_dim
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = nn.Linear(input_dim, 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        set_global_seed(self.seed)
        self.net.train()

        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        pos = float((y_t == 1).sum().item())
        neg = float((y_t == 0).sum().item())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X.astype(np.float32)))
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        return probs.astype(np.float32)

    def get_params(self) -> Dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    def state_dict_for_save(self) -> Dict[str, object]:
        return {
            "params": self.get_params(),
            "state_dict": self.net.state_dict(),
        }


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class TorchLSTMModel:
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        seed: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 80,
        batch_size: int = 128,
    ):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = _LSTMNet(input_dim, hidden_dim, num_layers, dropout)

    def fit(self, seq_x: np.ndarray, y: np.ndarray) -> None:
        set_global_seed(self.seed)
        self.net.train()

        X_t = torch.from_numpy(seq_x.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        pos = float((y_t == 1).sum().item())
        neg = float((y_t == 0).sum().item())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

    def predict_proba(self, seq_x: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(seq_x.astype(np.float32)))
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        return probs.astype(np.float32)

    def get_params(self) -> Dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    def state_dict_for_save(self) -> Dict[str, object]:
        return {
            "params": self.get_params(),
            "state_dict": self.net.state_dict(),
        }


@dataclass
class TreeNode:
    prob: float
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


class TorchDecisionTree:
    def __init__(
        self,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: Optional[int],
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root: Optional[TreeNode] = None

    @staticmethod
    def _gini(y: torch.Tensor) -> float:
        if y.numel() == 0:
            return 0.0
        p1 = float(y.float().mean().item())
        return 1.0 - p1 * p1 - (1.0 - p1) * (1.0 - p1)

    def _best_split(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        idx: torch.Tensor,
        generator: torch.Generator,
    ) -> Tuple[Optional[int], Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        n_features = X.shape[1]
        m_try = self.max_features if self.max_features is not None else max(1, int(math.sqrt(n_features)))
        m_try = max(1, min(m_try, n_features))

        feat_idx = torch.randperm(n_features, generator=generator)[:m_try]

        best_gini = float("inf")
        best_feat = None
        best_thr = None
        best_left = None
        best_right = None

        parent_count = len(idx)
        if parent_count < self.min_samples_split:
            return None, None, None, None

        for f in feat_idx.tolist():
            values = X[idx, f]
            if values.numel() < 2:
                continue

            quantiles = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
            thresholds = torch.quantile(values, quantiles).unique()

            for thr in thresholds.tolist():
                left_mask = values <= float(thr)
                right_mask = ~left_mask
                left_idx = idx[left_mask]
                right_idx = idx[right_mask]

                if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                    continue

                left_gini = self._gini(y[left_idx])
                right_gini = self._gini(y[right_idx])
                weighted = (len(left_idx) / parent_count) * left_gini + (len(right_idx) / parent_count) * right_gini

                if weighted < best_gini:
                    best_gini = weighted
                    best_feat = int(f)
                    best_thr = float(thr)
                    best_left = left_idx
                    best_right = right_idx

        return best_feat, best_thr, best_left, best_right

    def _build(self, X: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, depth: int, generator: torch.Generator) -> TreeNode:
        y_sub = y[idx]
        prob = float(y_sub.float().mean().item()) if len(idx) else 0.0

        if (
            depth >= self.max_depth
            or len(idx) < self.min_samples_split
            or len(torch.unique(y_sub)) <= 1
        ):
            return TreeNode(prob=prob)

        feat, thr, left_idx, right_idx = self._best_split(X, y, idx, generator)
        if feat is None or thr is None or left_idx is None or right_idx is None:
            return TreeNode(prob=prob)

        left = self._build(X, y, left_idx, depth + 1, generator)
        right = self._build(X, y, right_idx, depth + 1, generator)
        return TreeNode(prob=prob, feature_idx=feat, threshold=thr, left=left, right=right)

    def fit(self, X: torch.Tensor, y: torch.Tensor, generator: torch.Generator) -> None:
        idx = torch.arange(X.shape[0], dtype=torch.long)
        self.root = self._build(X, y, idx, depth=0, generator=generator)

    def _predict_one(self, x: torch.Tensor, node: TreeNode) -> float:
        cur = node
        while cur.feature_idx is not None and cur.threshold is not None and cur.left is not None and cur.right is not None:
            if float(x[cur.feature_idx].item()) <= cur.threshold:
                cur = cur.left
            else:
                cur = cur.right
        return float(cur.prob)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        if self.root is None:
            raise RuntimeError("Decision tree is not fitted")
        probs = torch.zeros(X.shape[0], dtype=torch.float32)
        for i in range(X.shape[0]):
            probs[i] = self._predict_one(X[i], self.root)
        return probs

    @staticmethod
    def _node_to_dict(node: TreeNode) -> Dict[str, object]:
        return {
            "prob": node.prob,
            "feature_idx": node.feature_idx,
            "threshold": node.threshold,
            "left": TorchDecisionTree._node_to_dict(node.left) if node.left else None,
            "right": TorchDecisionTree._node_to_dict(node.right) if node.right else None,
        }

    def to_dict(self) -> Dict[str, object]:
        if self.root is None:
            raise RuntimeError("Decision tree is not fitted")
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "root": self._node_to_dict(self.root),
        }


class TorchRandomForestModel:
    def __init__(
        self,
        seed: int,
        n_estimators: int = 120,
        max_depth: int = 10,
        min_samples_split: int = 8,
        min_samples_leaf: int = 3,
        max_features: Optional[int] = None,
    ):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees: List[TorchDecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        set_global_seed(self.seed)

        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))

        n = X_t.shape[0]
        g = torch.Generator().manual_seed(self.seed)
        self.trees = []

        for _ in range(self.n_estimators):
            sample_idx = torch.randint(0, n, (n,), generator=g)
            tree = TorchDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
            tree.fit(X_t[sample_idx], y_t[sample_idx], generator=g)
            self.trees.append(tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X.astype(np.float32))
        if not self.trees:
            raise RuntimeError("Random forest is not fitted")

        all_probs = []
        for tree in self.trees:
            all_probs.append(tree.predict_proba(X_t))
        stacked = torch.stack(all_probs, dim=0)
        return stacked.mean(dim=0).cpu().numpy().astype(np.float32)

    def get_params(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
        }

    def state_dict_for_save(self) -> Dict[str, object]:
        return {
            "params": self.get_params(),
            "trees": [tree.to_dict() for tree in self.trees],
        }


def _normalize_importance(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    total = float(arr.sum())
    if total > 0:
        arr = arr / total
    return arr.astype(np.float32)


def _tree_split_counts(node: Optional[TreeNode], counts: np.ndarray) -> None:
    if node is None:
        return
    if node.feature_idx is not None and 0 <= int(node.feature_idx) < int(len(counts)):
        counts[int(node.feature_idx)] += 1.0
    _tree_split_counts(node.left, counts)
    _tree_split_counts(node.right, counts)


def _rows_from_importance(
    *,
    model_name: str,
    method: str,
    dataset_name: str,
    seed: int,
    feature_names: Sequence[str],
    raw_importance: np.ndarray,
    n_reference_rows: int,
    n_effective_rows: int,
) -> List[Dict[str, object]]:
    normalized = _normalize_importance(raw_importance)
    rows: List[Dict[str, object]] = []
    for i, feat in enumerate(feature_names):
        rows.append(
            {
                "seed": int(seed),
                "dataset": str(dataset_name),
                "model": str(model_name),
                "feature": str(feat),
                "importance_raw": float(raw_importance[i]) if i < len(raw_importance) else 0.0,
                "importance_norm": float(normalized[i]) if i < len(normalized) else 0.0,
                "method": str(method),
                "n_reference_rows": int(n_reference_rows),
                "n_effective_rows": int(n_effective_rows),
            }
        )
    return rows


def compute_feature_importance(
    *,
    preprocessor: TabularPreprocessor,
    fitted_models: Dict[str, object],
    reference_df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    selected_models: Sequence[str],
    lstm_sequence_length: int,
    seed: int,
    dataset_name: str = "chicago_2025",
    max_lstm_samples: int = 2048,
) -> pd.DataFrame:
    selected = normalize_model_selection(selected_models)
    feature_cols = list(numeric_features) + list(categorical_features)
    if not feature_cols or reference_df.empty:
        return pd.DataFrame(
            columns=[
                "seed",
                "dataset",
                "model",
                "feature",
                "importance_raw",
                "importance_norm",
                "method",
                "n_reference_rows",
                "n_effective_rows",
            ]
        )

    X_tab = preprocessor.transform(reference_df[feature_cols])
    feature_names = preprocessor.output_feature_names if preprocessor.output_feature_names else list(feature_cols)
    n_features = int(len(feature_names))
    if n_features == 0:
        return pd.DataFrame(
            columns=[
                "seed",
                "dataset",
                "model",
                "feature",
                "importance_raw",
                "importance_norm",
                "method",
                "n_reference_rows",
                "n_effective_rows",
            ]
        )

    rows: List[Dict[str, object]] = []

    if "baseline_persistence" in selected:
        raw = np.zeros(n_features, dtype=np.float32)
        if "lag_0" in feature_names:
            raw[feature_names.index("lag_0")] = 1.0
        elif n_features > 0:
            raw[0] = 1.0
        rows.extend(
            _rows_from_importance(
                model_name="baseline_persistence",
                method="rule_based",
                dataset_name=dataset_name,
                seed=seed,
                feature_names=feature_names,
                raw_importance=raw,
                n_reference_rows=len(reference_df),
                n_effective_rows=len(reference_df),
            )
        )

    if "logistic_regression" in selected and "logistic_regression" in fitted_models:
        model = fitted_models["logistic_regression"]
        w = model.net.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
        raw = np.abs(w)
        rows.extend(
            _rows_from_importance(
                model_name="logistic_regression",
                method="coef_abs",
                dataset_name=dataset_name,
                seed=seed,
                feature_names=feature_names,
                raw_importance=raw,
                n_reference_rows=len(reference_df),
                n_effective_rows=len(reference_df),
            )
        )

    if "random_forest" in selected and "random_forest" in fitted_models:
        rf = fitted_models["random_forest"]
        split_counts = np.zeros(n_features, dtype=np.float32)
        for tree in rf.trees:
            _tree_split_counts(tree.root, split_counts)
        rows.extend(
            _rows_from_importance(
                model_name="random_forest",
                method="split_count",
                dataset_name=dataset_name,
                seed=seed,
                feature_names=feature_names,
                raw_importance=split_counts,
                n_reference_rows=len(reference_df),
                n_effective_rows=len(reference_df),
            )
        )

    if "lstm" in selected and "lstm" in fitted_models:
        lstm = fitted_models["lstm"]
        seq_x, _, _ = build_lstm_sequences(
            df=reference_df,
            X_tab=X_tab,
            sequence_length=lstm_sequence_length,
            include_target=False,
        )
        n_effective = int(len(seq_x))
        if n_effective > 0:
            seq_sample = seq_x
            if n_effective > int(max_lstm_samples):
                rng = np.random.default_rng(int(seed))
                sample_idx = rng.choice(n_effective, size=int(max_lstm_samples), replace=False)
                seq_sample = seq_x[sample_idx]

            x_t = torch.from_numpy(seq_sample.astype(np.float32))
            x_t.requires_grad_(True)
            lstm.net.eval()
            lstm.net.zero_grad()
            logits = lstm.net(x_t)
            target = logits.mean()
            target.backward()
            if x_t.grad is not None:
                raw = (
                    x_t.grad.detach()
                    .abs()
                    .mean(dim=(0, 1))
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            else:
                raw = np.zeros(n_features, dtype=np.float32)

            rows.extend(
                _rows_from_importance(
                    model_name="lstm",
                    method="grad_abs_mean",
                    dataset_name=dataset_name,
                    seed=seed,
                    feature_names=feature_names,
                    raw_importance=raw,
                    n_reference_rows=len(reference_df),
                    n_effective_rows=n_effective,
                )
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "seed",
                "dataset",
                "model",
                "feature",
                "importance_raw",
                "importance_norm",
                "method",
                "n_reference_rows",
                "n_effective_rows",
            ]
        )
    return out


def fit_models(
    chicago_train_xy: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_seed: int,
    selected_models: Sequence[str],
    lstm_sequence_length: int,
) -> Tuple[TabularPreprocessor, Dict[str, object], List[Dict[str, object]]]:
    selected = normalize_model_selection(selected_models)

    feature_cols = list(numeric_features) + list(categorical_features)
    X_raw = chicago_train_xy[feature_cols]
    y_train = chicago_train_xy["y"].astype(int).to_numpy()

    preprocessor = TabularPreprocessor(numeric_features=numeric_features, categorical_features=categorical_features)
    t0 = time.perf_counter()
    X_tab = preprocessor.fit_transform(X_raw)
    preproc_seconds = time.perf_counter() - t0

    fitted_models: Dict[str, object] = {}
    train_logs: List[Dict[str, object]] = [
        {
            "model": "preprocessor",
            "train_seconds": preproc_seconds,
            "n_train_rows": int(len(chicago_train_xy)),
            "n_features_output": int(X_tab.shape[1]),
            "params": preprocessor.to_dict(),
        }
    ]

    if "logistic_regression" in selected:
        model = TorchLogisticRegressionModel(input_dim=X_tab.shape[1], seed=random_seed)
        start = time.perf_counter()
        model.fit(X_tab, y_train)
        elapsed = time.perf_counter() - start
        fitted_models["logistic_regression"] = model
        train_logs.append(
            {
                "model": "logistic_regression",
                "train_seconds": elapsed,
                "n_train_rows": int(len(chicago_train_xy)),
                "n_features_output": int(X_tab.shape[1]),
                "params": model.get_params(),
            }
        )

    if "random_forest" in selected:
        rf = TorchRandomForestModel(seed=random_seed)
        start = time.perf_counter()
        rf.fit(X_tab, y_train)
        elapsed = time.perf_counter() - start
        fitted_models["random_forest"] = rf
        train_logs.append(
            {
                "model": "random_forest",
                "train_seconds": elapsed,
                "n_train_rows": int(len(chicago_train_xy)),
                "n_features_output": int(X_tab.shape[1]),
                "params": rf.get_params(),
            }
        )

    if "lstm" in selected:
        seq_x, seq_y, seq_row_idx = build_lstm_sequences(
            df=chicago_train_xy,
            X_tab=X_tab,
            sequence_length=lstm_sequence_length,
            include_target=True,
        )
        if len(seq_x) != len(chicago_train_xy):
            raise ValueError(
                "LSTM training sequences are not aligned with training rows. "
                f"Expected {len(chicago_train_xy)}, got {len(seq_x)}."
            )
        model = TorchLSTMModel(
            input_dim=X_tab.shape[1],
            sequence_length=lstm_sequence_length,
            seed=random_seed,
        )
        start = time.perf_counter()
        if len(seq_x) == 0:
            raise ValueError(
                f"No LSTM training sequences generated. Reduce --lstm_sequence_length (current: {lstm_sequence_length})."
            )
        model.fit(seq_x, seq_y)
        elapsed = time.perf_counter() - start
        fitted_models["lstm"] = model
        train_logs.append(
            {
                "model": "lstm",
                "train_seconds": elapsed,
                "n_train_rows": int(len(chicago_train_xy)),
                "n_sequence_rows": int(len(seq_x)),
                "sequence_length": int(lstm_sequence_length),
                "n_features_output": int(X_tab.shape[1]),
                "params": model.get_params(),
            }
        )

    if "baseline_persistence" in selected:
        train_logs.append(
            {
                "model": "baseline_persistence",
                "train_seconds": 0.0,
                "n_train_rows": int(len(chicago_train_xy)),
                "n_features_output": int(X_tab.shape[1]),
                "params": {
                    "rule": "predict y(t+1) from lag_0 threshold",
                },
            }
        )

    return preprocessor, fitted_models, train_logs


def predict_with_all_models(
    df: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    fitted_models: Dict[str, object],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    persistence_threshold: Union[int, Mapping[str, int]],
    dataset_name: str,
    selected_models: Sequence[str],
    lstm_sequence_length: int,
) -> pd.DataFrame:
    selected = normalize_model_selection(selected_models)
    feature_cols = list(numeric_features) + list(categorical_features)
    X_tab = preprocessor.transform(df[feature_cols])

    outputs: List[pd.DataFrame] = []
    base_cols = ["region_id", "time_id", "y", "y_count"]
    if "crime_group" in df.columns:
        base_cols.insert(2, "crime_group")

    if "baseline_persistence" in selected:
        baseline = df[base_cols].copy()
        baseline["model"] = "baseline_persistence"
        baseline["dataset"] = dataset_name
        baseline["y_score"] = df["lag_0"].astype(float)
        if isinstance(persistence_threshold, Mapping):
            if "crime_group" in baseline.columns:
                threshold_map = {str(k): int(v) for k, v in persistence_threshold.items()}
                default_threshold = int(max([1] + list(threshold_map.values())))
                threshold_series = baseline["crime_group"].astype(str).map(threshold_map).fillna(default_threshold).astype(float)
                baseline["y_pred"] = (baseline["y_score"] >= threshold_series).astype(int)
            else:
                default_threshold = int(max([1] + [int(v) for v in persistence_threshold.values()]))
                baseline["y_pred"] = (baseline["y_score"] >= float(default_threshold)).astype(int)
        elif isinstance(persistence_threshold, Integral):
            baseline["y_pred"] = (baseline["y_score"] >= int(persistence_threshold)).astype(int)
        else:
            raise ValueError("persistence_threshold must be int or mapping[str, int]")
        outputs.append(baseline)

    if "logistic_regression" in selected and "logistic_regression" in fitted_models:
        model = fitted_models["logistic_regression"]
        pred_df = df[base_cols].copy()
        pred_df["model"] = "logistic_regression"
        pred_df["dataset"] = dataset_name
        pred_df["y_score"] = model.predict_proba(X_tab)
        pred_df["y_pred"] = (pred_df["y_score"] >= 0.5).astype(int)
        outputs.append(pred_df)

    if "random_forest" in selected and "random_forest" in fitted_models:
        model = fitted_models["random_forest"]
        pred_df = df[base_cols].copy()
        pred_df["model"] = "random_forest"
        pred_df["dataset"] = dataset_name
        pred_df["y_score"] = model.predict_proba(X_tab)
        pred_df["y_pred"] = (pred_df["y_score"] >= 0.5).astype(int)
        outputs.append(pred_df)

    if "lstm" in selected and "lstm" in fitted_models:
        model = fitted_models["lstm"]
        seq_x, _, row_idx = build_lstm_sequences(
            df=df,
            X_tab=X_tab,
            sequence_length=lstm_sequence_length,
            include_target=False,
        )
        unique_row_count = int(np.unique(row_idx).size)
        if unique_row_count != len(df):
            raise ValueError(
                "LSTM prediction rows are not fully aligned with dataset rows. "
                f"Expected unique rows={len(df)}, got {unique_row_count}."
            )
        if len(seq_x) > 0:
            pred_df = df.reset_index(drop=True).iloc[row_idx][base_cols].copy()
            pred_df["model"] = "lstm"
            pred_df["dataset"] = dataset_name
            pred_df["y_score"] = model.predict_proba(seq_x)
            pred_df["y_pred"] = (pred_df["y_score"] >= 0.5).astype(int)
            outputs.append(pred_df)

    if not outputs:
        raise ValueError("No predictions produced. Check selected models and fitted models.")

    return pd.concat(outputs, ignore_index=True)


def persist_models(
    output_dir: Path,
    preprocessor: TabularPreprocessor,
    fitted_models: Dict[str, object],
    train_logs: List[Dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale artifacts from previous runs to avoid mixing model families.
    for old in output_dir.glob("*.joblib"):
        old.unlink()
    for old in output_dir.glob("*.pt"):
        old.unlink()

    (output_dir / "preprocessor.json").write_text(
        json.dumps(preprocessor.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    for name, model in fitted_models.items():
        payload = model.state_dict_for_save()
        torch.save(payload, output_dir / f"{name}.pt")

    (output_dir / "training_log.json").write_text(
        json.dumps(train_logs, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone PyTorch model training script.")
    parser.add_argument("--train_xy", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--numeric", nargs="+", required=True)
    parser.add_argument("--categorical", nargs="*", default=[])
    parser.add_argument("--models", nargs="+", default=["all"], help="all | baseline | logistic | rf | lstm")
    parser.add_argument("--lstm_sequence_length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_xy.suffix.lower() == ".parquet":
        train_xy = pd.read_parquet(args.train_xy)
    else:
        train_xy = pd.read_csv(args.train_xy)

    selected = normalize_model_selection(args.models)
    preprocessor, models, logs = fit_models(
        chicago_train_xy=train_xy,
        numeric_features=args.numeric,
        categorical_features=args.categorical,
        random_seed=args.seed,
        selected_models=selected,
        lstm_sequence_length=args.lstm_sequence_length,
    )
    persist_models(args.model_dir, preprocessor, models, logs)


if __name__ == "__main__":
    main()
