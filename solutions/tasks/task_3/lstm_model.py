import torch
import torch.nn as nn
import numpy as np


class LSTMForecast(nn.Module):
    """
    Rozszerzona wersja LSTM dla prognozowania szeregu czasowego.

    input_dim  : liczba cech wejściowych na jeden krok czasowy (u nas 1, bo to transaction_count).
    hidden_dim : liczba jednostek w warstwie ukrytej.
    num_layers : liczba warstw LSTM.
    output_dim : liczba wartości na wyjściu (u nas 1, bo chcemy przewidzieć pojedynczą wartość).
    dropout    : dropout między warstwami LSTM.
    """

    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        batch_size = x.size(0)
        # Stan początkowy LSTM (h0, c0). Tutaj inicjalizowane zerami.
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_dim)
        out = out[:, -1, :]  # Pobieramy wyjście z ostatniego kroku czasowego
        out = self.fc(out)  # (batch_size, output_dim)
        return out


def create_sequences(data: np.ndarray, seq_length: int = 7):
    """
    Generuje sekwencje (X, y) do trenowania modelu sekwencyjnego.
    data: np.ndarray 1D z wartościami np. transakcji dziennych
    seq_length: długość okna czasowego, np. 7 dni
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x_seq = data[i: i + seq_length]
        y_seq = data[i + seq_length]  # wartość przewidywana (kolejny dzień)
        xs.append(x_seq)
        ys.append(y_seq)

    return np.array(xs), np.array(ys)


def create_data_loader(data: np.ndarray,
                       seq_length: int = 7,
                       batch_size: int = 32,
                       shuffle: bool = True):
    """
    Tworzy obiekt DataLoader dla danych sekwencyjnych.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    X, y = create_sequences(data, seq_length)
    # Dodajemy wymiar cechy (input_dim=1)
    X_tensor = torch.from_numpy(X).float().unsqueeze(-1)  # (samples, seq_length, 1)
    y_tensor = torch.from_numpy(y).float().unsqueeze(-1)  # (samples, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_lstm(model: nn.Module,
               train_loader,
               val_loader=None,
               epochs: int = 50,
               lr: float = 1e-3,
               device: str = 'cpu'):
    """
    Trenuje model LSTM przy użyciu Adam + MSELoss, z opcjonalną walidacją
    i wczesnym zatrzymaniem (early stopping).

    train_loader: DataLoader dla zbioru treningowego
    val_loader  : DataLoader dla zbioru walidacyjnego
    epochs      : liczba epok
    lr          : learning rate
    device      : 'cpu' lub 'cuda'
    """
    import torch.optim as optim
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5  # Maksymalna liczba epok bez poprawy do przerwania

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    val_out = model(X_val)
                    val_loss = criterion(val_out, y_val)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)

            print(f"Epoch [{epoch}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")

            # Wczesne zatrzymanie (early stopping)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        else:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}")


def multi_step_forecast(model: nn.Module,
                        init_seq: np.ndarray,
                        forecast_horizon: int = 90,
                        device: str = 'cpu'):
    """
    Wykonuje iteracyjną prognozę:
    - init_seq: ostatnie seq_length wartości (numpy array) służące jako punkt startowy
    - forecast_horizon: liczba kolejnych kroków (dni) do prognozy
    - Zwraca wektor z przewidywaną wartością na każdy z tych dni.
    """
    model.eval()
    seq_length = len(init_seq)
    forecast = []
    current_seq = init_seq.copy()

    with torch.no_grad():
        for _ in range(forecast_horizon):
            X = torch.from_numpy(current_seq).float().unsqueeze(0).unsqueeze(-1).to(device)
            out = model(X)
            next_val = out.item()
            forecast.append(next_val)
            # "Przesuwamy" okno, usuwając najstarszy dzień i dokładając nowo przewidziany
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = next_val

    return np.array(forecast)
