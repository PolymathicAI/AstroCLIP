import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm


class MLP(nn.Module):
    """A simple feedforward neural network with 3 hidden layers."""

    def __init__(
        self, input_dim: int, num_classes: int, hidden_dim: int, dropout_rate: int
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.layers(x)
        x = self.softmax(x)
        return x.squeeze()


def train_eval_on_question(
    X_train: torch.tensor,
    X_test: torch.tensor,
    y_train: torch.tensor,
    y_test: torch.tensor,
    embed_dim: int,
    num_classes: int,
    MLP_dim: int = 128,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 25,
    dropout: float = 0.2,
) -> dict:
    """Function to train and evaluate a simple feedforward neural network on a dataset."""
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create a DataLoader
    samples_weight = y_train.max(dim=1).values  # Taking max fraction as the weight
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(samples_weight, len(samples_weight)),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Set up model
    mlp = MLP(embed_dim, num_classes, MLP_dim, dropout).cuda()
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-label classification
    optimizer = optim.Adam(mlp.parameters(), lr=lr)

    # Training loop
    best_val_loss = float("inf")
    best_metrics = None
    for epoch in epochs:
        mlp.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = mlp(data.cuda())
            loss = criterion(output.squeeze(), target.squeeze().cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        mlp.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = mlp(data.cuda())
                loss = criterion(output.squeeze(), target.squeeze().cuda())
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = mlp.state_dict()

    # Get the best  model
    mlp.load_state_dict(best_model)
    y_pred = mlp(X_test.cuda()).detach().cpu()

    # Discretize the predictions
    y_pred = (y_pred == torch.max(y_pred, dim=1, keepdim=True).values).int()
    y_true = (y_test == torch.max(y_test, dim=1, keepdim=True).values).int()

    # Compute and return the metrics
    accuracy = accuracy_score(y_true.numpy(), y_pred.numpy())
    f1_score = precision_recall_fscore_support(
        y_true.numpy(), y_pred.numpy(), average="weighted", zero_division=0
    )[2]
    return {"Accuracy": accuracy, "F1 Score": f1_score}
