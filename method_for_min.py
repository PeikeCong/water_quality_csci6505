import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.pruners import HyperbandPruner
import matplotlib.pyplot as plt

from preprocess import *

# Hyperparamters
SEQUENCE_SIZE = 8 
# HIDDEN_SIZE = 128 
HIDDEN_SIZE = 16
NUM_LAYERS = 2 
DROPOUT = 0.2 
BATCH_SIZE = 64
LEARNING_RATE = 1e-4 
EPOCH = 20 

# SEQUENCE_SIZE = 32
# HIDDEN_SIZE = 64
# NUM_LAYERS = 1
# DROPOUT = 0.5
# BATCH_SIZE = 32
# LEARNING_RATE = 0.0030597322902911482
# EPOCH = 10

class WaterQualityDataset(Dataset):
    def __init__(self, feature_dataset, target_dataset, sequence_length=16):
        self.feature_dataset = feature_dataset
        self.target_dataset = target_dataset
        self.sequence_length = sequence_length
        self.features, self.targets = self.preprocess_data()
        self.x, self.y = self.create_sequences()

    def preprocess_data(self):
        features = self.feature_dataset.values
        targets = self.target_dataset[['Dissolved Oxygen', 'pH', 'Turbidity']].values
        # normalize targets
        scaler = StandardScaler()
        targets = scaler.fit_transform(targets)
        return features, targets

    def create_sequences(self):
        x, y = [], []

        for i in range(self.sequence_length, len(self.feature_dataset)):
            x.append(self.features[i - self.sequence_length:i])
            y.append(self.targets[i])

        x = np.array(x)
        y = np.array(y)

        # print(len(x))
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def get_data(self, test_size=0.1, validation_size=0.1):
        total_size = len(self.x)
        train_size = int(total_size * (1 - test_size - validation_size))
        validation_size = int(total_size * validation_size)

        X_train = self.x[ : train_size]
        X_val = self.x[train_size : train_size + validation_size]
        X_test = self.x[train_size + validation_size : ]

        y_train = self.y[ : train_size]
        y_val = self.y[train_size : train_size + validation_size]
        y_test = self.y[train_size + validation_size : ]

        return X_train, X_val, X_test, y_train, y_val, y_test


class LSTMModel(nn.Module):
    def __init__(self, hidden_size = 64, num_layers = 2, dropout = 0.2):
        super(LSTMModel, self).__init__()
        self.input_size = 8
        self.output_size = 3
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,
                            num_layers = self.num_layers, batch_first = True,
                            dropout = self.dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, self.output_size),
            nn.ReLU(),                              
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc_layers(hn[-1])
        return out

class ImprovedLSTMModel(nn.Module):
    def __init__(self, hidden_size = 64, num_layers = 2, dropout = 0.2):
        super(ImprovedLSTMModel, self).__init__()
        self.input_size = 8
        self.output_size = 3
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,
                            num_layers = self.num_layers, batch_first = True,
                            dropout = self.dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),                              
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.fc_out = nn.Linear(hidden_size // 4, self.output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc_layers(hn[-1])
        out = self.fc_out(out)
        return out


class RNNModel(nn.Module):
    def __init__(self, hidden_size = 64, num_layers = 2, dropout = 0.2):
        super(RNNModel, self).__init__()
        self.input_size = 8
        self.output_size = 3
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.rnn = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size,
                            num_layers = self.num_layers, batch_first = True,
                            dropout = self.dropout)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, self.output_size),
            nn.ReLU(),                              
        )

    def forward(self, x):
        _, hn = self.rnn(x)
        out = self.fc_layers(hn[-1])
        return out


class TimeSeriesTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout = 0.1, nhead = 8):
        super(TimeSeriesTransformer, self).__init__()
        self.input_size = 8
        self.output_size = 3
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        self.positional_encoding = PositionalEncoding(self.hidden_size, self.dropout)
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model = self.hidden_size, nhead=nhead, dropout = self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers = self.num_layers)
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        # convert to (sequence_length, batch_size, hidden_size) for transformer
        x = x.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(x)
        last_output = transformer_out[-1, :, :]
        output = self.fc(last_output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.2, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


def train_model(model, train_loader, val_loader, lr = 0.001, epochs=20):
    train_losses = []
    val_losses = []
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    patience = 5
    best_val_loss = float('inf') 
    epochs_without_improvement = 0
    early_stop_epoch = -1

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs, targets
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs, targets
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            early_stop_epoch = epoch + 1
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    if early_stop_epoch != -1:
        epochs = early_stop_epoch

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    return train_losses[-1], val_losses[-1]
    


def objective(trial):
    sequence_length = trial.suggest_categorical('sequence_length', [8, 16, 32, 64])
    hidden_size = trial.suggest_int('hidden_size', 16, 32, step=8)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    epochs = trial.suggest_categorical('epochs', [10, 20])

    # data
    feature_dataset = read_data(PATH_MINI_KNN_PCA)
    target_dataset = read_data(PATH_MINI_KNN)
    water_dataset = WaterQualityDataset(feature_dataset, target_dataset, sequence_length)
    X_train, X_val, X_test, y_train, y_val, y_test = water_dataset.get_data(test_size=0.1, validation_size=0.1)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False)

    # model
    model = LSTMModel(hidden_size = hidden_size, num_layers = num_layers, dropout = dropout)
    # model = ImprovedLSTMModel(hidden_size = hidden_size, num_layers = num_layers, dropout = dropout)
    # model = RNNModel(hidden_size = hidden_size, num_layers = num_layers, dropout = dropout)
    # model = TimeSeriesTransformer(hidden_size = hidden_size, num_layers = num_layers, dropout = dropout)

    # train
    last_train_loss, last_val_loss = train_model(model, train_loader, val_loader, lr = learning_rate, epochs = epochs)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    test_loss = test_model(model, test_loader)

    return test_loss


def tune_hyperparameter():
    study = optuna.create_study(direction='minimize', pruner=HyperbandPruner())
    study.optimize(objective, n_trials = 10)
    best_params = study.best_params
    print("Best hyperparameters: ", best_params)



def test_model(model, test_loader):
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs, targets
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    return test_loss/len(test_loader)




def train_and_test():
    # get train, val, test data
    feature_dataset = read_data(PATH_MINI_KNN_PCA)
    target_dataset = read_data(PATH_MINI_KNN)
    water_dataset = WaterQualityDataset(feature_dataset, target_dataset, SEQUENCE_SIZE)
    X_train, X_val, X_test, y_train, y_val, y_test = water_dataset.get_data(test_size=0.1, validation_size=0.1)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")

    # create data loader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE, shuffle = False)

    # create model
    # model = LSTMModel(hidden_size = HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT)
    model = ImprovedLSTMModel(hidden_size = HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT)
    # model = RNNModel(hidden_size = HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT)
    # model = TimeSeriesTransformer(hidden_size = HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = DROPOUT)

    # train
    train_model(model, train_loader, val_loader, lr = LEARNING_RATE, epochs = EPOCH)

    # test
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    test_model(model, test_loader)



if __name__ == "__main__":
    train_and_test()
    # tune_hyperparameter()
