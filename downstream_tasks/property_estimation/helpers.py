import numpy as np
import torch
from astropy.table import Table
from numpy import ndarray
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from tqdm import tqdm


def few_shot(
    model: nn.Module,
    X_train: ndarray,
    y_train: ndarray,
    X_test: ndarray,
    max_epochs: int = 10,
    lr: float = 1e-3,
) -> ndarray:
    """Train a few-shot model using a simple neural network"""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Set up the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.cuda()
    model.train()
    for epoch in tqdm(range(max_epochs), desc="Training few-shot model"):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Make predictions
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu().numpy()
    return preds


def zero_shot(
    X_train: ndarray, y_train: ndarray, X_test: ndarray, n_neighbors: int = 64
) -> ndarray:
    """Train a zero-shot model using KNN"""
    print("Training zero-shot model")
    neigh = KNeighborsRegressor(weights="distance", n_neighbors=64)
    neigh.fit(X_train, y_train)
    preds = neigh.predict(X_test)
    return preds
