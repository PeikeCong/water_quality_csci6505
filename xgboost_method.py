import torch
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader, TensorDataset

from preprocess import *

# Hyperparamters
SEQUENCE_SIZE = 8 
BATCH_SIZE = 64

class WaterQualityDataset(Dataset):
    def __init__(self, feature_dataset, target_dataset, sequence_length=16):
        self.feature_dataset = feature_dataset
        self.target_dataset = target_dataset
        self.sequence_length = sequence_length
        self.features, self.targets = self.preprocess_data()
        self.x, self.y = self.create_sequences()

    def preprocess_data(self):
        features = self.feature_dataset.values
        targets = self.target_dataset[['Timestamp', 'Dissolved Oxygen', 'pH', 'Turbidity']].values
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


def train_and_test():
    # get train, val, test data
    feature_dataset = read_data(PATH_KNN_PCA)
    target_dataset = read_data(PATH_KNN)
    water_dataset = WaterQualityDataset(feature_dataset, target_dataset, SEQUENCE_SIZE)
    X_train, X_val, X_test, y_train, y_val, y_test = water_dataset.get_data(test_size=0.1, validation_size=0.1)
    # print(f"X_train shape: {X_train.shape}")
    # print(f"X_val shape: {X_val.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"y_val shape: {y_val.shape}")
    # print(f"y_test shape: {y_test.shape}")

    # create data loader
    # train_dataset = TensorDataset(X_train, y_train)
    # val_dataset = TensorDataset(X_val, y_val)
    # train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    # val_loader = DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE, shuffle = False)

    X_train_np = X_train.reshape(X_train.shape[0], -1).numpy()  # (24484, 8*9)
    X_val_np = X_val.reshape(X_val.shape[0], -1).numpy()        # (3060, 8*9)
    X_test_np = X_test.reshape(X_test.shape[0], -1).numpy()     # (3062, 8*9)
    y_train_np = y_train.numpy()  # (24484, 4)
    y_val_np = y_val.numpy()      # (3060, 4)
    y_test_np = y_test.numpy()    # (3062, 4)


    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    model.set_params(eval_metric="rmse")

    
    model.fit(
        X_train_np, y_train_np,                   
        eval_set=[(X_train_np, y_train_np),
                (X_val_np, y_val_np)],
        verbose=True
    )
    
    
    # y_val_pred = model.predict(X_val_np)
    y_test_pred = model.predict(X_test_np)

    # mse = mean_squared_error(y_val_np, y_val_pred)
    # print(f"Mean Squared Error: {mse:.2f}")
    mse = mean_squared_error(y_test_np, y_test_pred)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse:.2f}")




if __name__ == "__main__":
    train_and_test()