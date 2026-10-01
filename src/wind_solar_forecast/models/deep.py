"""PyTorch Deep Temporal Fusion architecture with multi-quantile loss and seed averaging."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) module from Temporal Fusion Transformer (Lim et al. 2021)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.skip = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        g = self.sigmoid(self.gate(h))
        return self.layer_norm(residual + g * h)


class TemporalFusionNet(nn.Module):
    """Deep temporal architecture combining GRN variable transformations, Bi-LSTM, and quantile output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_quantiles: int = 7,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.grn_input = GatedResidualNetwork(input_dim, hidden_dim, dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.grn_post = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        self.quantile_head = nn.Linear(hidden_dim, num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x

        h_grn = self.grn_input(x_seq)
        lstm_out, _ = self.lstm(h_grn)
        h_post = self.grn_post(lstm_out[:, -1, :])
        quantiles = self.quantile_head(h_post)
        return quantiles


def multi_quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Computes pinball (quantile) loss across all target quantile levels.

    Loss = sum_q max(q * (y - y_hat), (1 - q) * (y_hat - y))
    """
    # preds: (batch_size, n_quantiles), target: (batch_size, 1)
    err = target - preds
    loss = torch.max(quantiles * err, (quantiles - 1.0) * err)
    return torch.mean(loss)


class DeepProbabilisticForecaster:
    """Ensemble of Deep Temporal Fusion Nets trained across multiple random seeds for robustness."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        hidden_dim: int = 64,
        max_epochs: int = 25,
        batch_size: int = 64,
        learning_rate: float = 0.002,
        n_seeds: int = 5,
        device: str = "cpu"
    ) -> None:
        self.quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_seeds = n_seeds
        self.device = torch.device(device)
        self.models: list[TemporalFusionNet] = []
        self.input_dim: int = 0
        self.feature_means: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "DeepProbabilisticForecaster":
        """Fits seed-averaged deep networks using multi-quantile pinball loss."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.input_dim = X_arr.shape[1]
        self.feature_means = np.mean(X_arr, axis=0)
        self.feature_stds = np.std(X_arr, axis=0) + 1e-6
        X_norm = (X_arr - self.feature_means) / self.feature_stds

        q_tensor = torch.tensor(self.quantiles, dtype=torch.float32, device=self.device)
        self.models = []

        for seed in range(self.n_seeds):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            model = TemporalFusionNet(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_quantiles=len(self.quantiles)
            ).to(self.device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

            dataset = TensorDataset(torch.from_numpy(X_norm).to(self.device), torch.from_numpy(y_arr).to(self.device))
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            model.train()
            for _ in range(self.max_epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    out = model(batch_x)
                    loss = multi_quantile_loss(out, batch_y, q_tensor)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            model.eval()
            self.models.append(model)

        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Averages predicted quantiles across all seeds and enforces monotonicity."""
        X_arr = np.asarray(X, dtype=np.float32)
        assert self.feature_means is not None and self.feature_stds is not None
        X_norm = (X_arr - self.feature_means) / self.feature_stds

        with torch.no_grad():
            tensor_x = torch.from_numpy(X_norm).to(self.device)
            seed_preds = []
            for model in self.models:
                raw_pred = model(tensor_x).cpu().numpy()
                seed_preds.append(raw_pred)

            avg_preds = np.mean(seed_preds, axis=0)
            avg_preds = np.clip(avg_preds, 0.0, 1.0)
            # Enforce non-crossing monotonicity
            avg_preds = np.sort(avg_preds, axis=1)

        return avg_preds

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Median quantile forecast (50th percentile)."""
        q_preds = self.predict_quantiles(X)
        mid_idx = len(self.quantiles) // 2
        return q_preds[:, mid_idx]
