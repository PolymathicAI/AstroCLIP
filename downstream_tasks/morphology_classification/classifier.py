import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, dropout_rate):
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

    def forward(self, x):
        x = self.layers(x)
        x = self.softmax(x)
        return x.squeeze()


def train_eval_MLP(
    X_train,
    X_test,
    y_train,
    y_test,
    embed_dim,
    num_classes,
    MLP_dim=128,
    lr=1e-3,
    epochs=25,
    dropout=0.2,
):
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create a DataLoader
    samples_weight = y_train.max(dim=1).values  # Taking max fraction as the weight

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        sampler=WeightedRandomSampler(samples_weight, len(samples_weight)),
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    mlp = MLP(embed_dim, num_classes, MLP_dim, dropout)
    criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
    optimizer = optim.Adam(mlp.parameters(), lr=lr)

    # Training loop
    best_val_loss = float("inf")
    best_metrics = None

    for epoch in range(epochs):  # Define your number of epochs
        mlp.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = mlp(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        mlp.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = mlp(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Report every 25 epochs
        if epoch % 25 == 0:
            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = mlp.state_dict()

    mlp.load_state_dict(best_model)
    y_pred = mlp(X_test).detach()

    y_pred = (y_pred == torch.max(y_pred, dim=1, keepdim=True).values).int()
    y_true = (y_test == torch.max(y_test, dim=1, keepdim=True).values).int()

    accuracy = accuracy_score(y_true.numpy(), y_pred.numpy())
    f1_score = precision_recall_fscore_support(
        y_true.numpy(), y_pred.numpy(), average="weighted", zero_division=0
    )[2]
    return {"Accuracy": accuracy, "F1 Score": f1_score}
