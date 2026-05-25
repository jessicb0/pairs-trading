import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


LSTM_FEATURES = [
    "Zscore",
    "Spread",
    "Return_A",
    "Return_B"
]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size=32,
        num_layers=1,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


def prepare_lstm_dataset(signals, sequence_length=20):
    df = signals.copy()

    for col in LSTM_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans signals : {col}")

    df["Target"] = (
        df["Zscore"].abs().shift(-1) < df["Zscore"].abs()
    ).astype(int)

    df = df.dropna()

    X_raw = df[LSTM_FEATURES].values
    y_raw = df["Target"].values
    dates = df.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_sequences = []
    y_sequences = []
    sequence_dates = []

    for i in range(sequence_length, len(X_scaled)):
        X_sequences.append(X_scaled[i - sequence_length:i])
        y_sequences.append(y_raw[i])
        sequence_dates.append(dates[i])

    return (
        np.array(X_sequences),
        np.array(y_sequences),
        pd.Index(sequence_dates),
        scaler
    )


def temporal_lstm_split(X, y, dates, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))

    return (
        X[:split_idx],
        X[split_idx:],
        y[:split_idx],
        y[split_idx:],
        dates[:split_idx],
        dates[split_idx:]
    )


def train_lstm_classifier(
    signals,
    sequence_length=20,
    test_size=0.2,
    epochs=50,
    batch_size=16,
    learning_rate=0.001,
    hidden_size=32,
    random_state=42
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X, y, dates, scaler = prepare_lstm_dataset(
        signals=signals,
        sequence_length=sequence_length
    )

    X_train, X_test, y_train, y_test, dates_train, dates_test = temporal_lstm_split(
        X,
        y,
        dates,
        test_size=test_size
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=False
    )

    model = LSTMClassifier(
        n_features=X_train.shape[2],
        hidden_size=hidden_size
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "loss": []
    }

    model.train()

    for epoch in range(epochs):
        epoch_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        history["loss"].append(np.mean(epoch_losses))

    model.eval()

    with torch.no_grad():
        test_logits = model(X_test_tensor)
        y_proba = torch.sigmoid(test_logits).numpy().flatten()

    y_pred = (y_proba > 0.5).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred
    )

    prediction_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Convergence_Probability": y_proba
    }, index=dates_test)

    latest_probability = y_proba[-1]

    return {
        "Model": model,
        "Scaler": scaler,
        "History": history,
        "Accuracy": accuracy,
        "Classification_Report": report,
        "Prediction_DF": prediction_df,
        "Latest_Probability": latest_probability,
        "Sequence_Length": sequence_length,
        "Features": LSTM_FEATURES
    }
    
class LSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size=16,
        num_layers=1,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


def prepare_lstm_regression_dataset(signals, sequence_length=5):
    df = signals.copy()

    for col in LSTM_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans signals : {col}")

    if "Strategy_Return" not in df.columns:
        raise ValueError(
            "La colonne Strategy_Return est manquante. "
            "Lance d'abord run_simple_backtest(signals)."
        )

    df["Target_Return_J1"] = df["Strategy_Return"].shift(-1)

    df = df.dropna()

    X_raw = df[LSTM_FEATURES].values
    y_raw = df["Target_Return_J1"].values
    dates = df.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_sequences = []
    y_sequences = []
    sequence_dates = []

    for i in range(sequence_length, len(X_scaled)):
        X_sequences.append(X_scaled[i - sequence_length:i])
        y_sequences.append(y_raw[i])
        sequence_dates.append(dates[i])

    return (
        np.array(X_sequences),
        np.array(y_sequences),
        pd.Index(sequence_dates),
        scaler
    )


def train_lstm_regressor(
    signals,
    sequence_length=5,
    test_size=0.2,
    epochs=10,
    batch_size=64,
    learning_rate=0.001,
    hidden_size=8,
    random_state=42
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X, y, dates, scaler = prepare_lstm_regression_dataset(
        signals=signals,
        sequence_length=sequence_length
    )

    X_train, X_test, y_train, y_test, dates_train, dates_test = temporal_lstm_split(
        X,
        y,
        dates,
        test_size=test_size
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=False
    )

    model = LSTMRegressor(
        n_features=X_train.shape[2],
        hidden_size=hidden_size
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "loss": []
    }

    model.train()

    for epoch in range(epochs):
        epoch_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            prediction = model(batch_X)
            loss = criterion(prediction, batch_y)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        history["loss"].append(np.mean(epoch_losses))
        print(f"LSTM Regressor Epoch {epoch + 1}/{epochs} - Loss: {history['loss'][-1]:.8f}")

    model.eval()

    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().flatten()

    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    if np.var(y_test) == 0:
        r2 = 0
    else:
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum(
            (y_test - np.mean(y_test)) ** 2
        )

    prediction_df = pd.DataFrame({
        "Actual_Return_J1": y_test,
        "Predicted_Return_J1_LSTM": y_pred
    }, index=dates_test)

    return {
        "Model": model,
        "Scaler": scaler,
        "History": history,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Prediction_DF": prediction_df,
        "Latest_Prediction": y_pred[-1],
        "Sequence_Length": sequence_length,
        "Features": LSTM_FEATURES
    }