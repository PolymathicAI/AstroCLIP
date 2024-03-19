from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader

dataset = load_from_disk("/mnt/ceph/users/polymathic/mmoma/datasets/astroclip_file/")
dataset.set_format(type="torch", columns=["spectrum"])


loader = {
    k: DataLoader(dataset[k], batch_size=128, shuffle=True) for k in ["train", "test"]
}

# preprocessing the samples
# This step, Z-scores each sample individually and encodes
# this information is stored in the first element of each sample
# Then the sample is then unfolded into overlapping regions
# as a continuous tokenization

import numpy as np


def slice(x, section_length=10, overlap=5):

    start_indices = np.arange(0, len(x) - overlap, section_length - overlap)
    sections = [x[start : start + section_length] for start in start_indices]

    # If the last section is not of length 'section_length', you can decide whether to keep or discard it
    if len(sections[-1]) < section_length:
        sections.pop(-1)  # Discard the last section

    return np.concatenate(sections, 1).T


def preprocess(samples):
    out = []

    for x in samples["spectrum"]:
        x = np.array(x)
        std, mean = x.std(), x.mean()
        # skipping samples that are all zero
        if std == 0:
            continue
        x = (x - mean) / std
        x = slice(x, 194, 97)
        x = np.pad(x, pad_width=((1, 0), (2, 0)), mode="constant", constant_values=0)

        x[0, 0] = (mean - 2) / 2
        x[0, 1] = (std - 2) / 8

        out.append(x)
    # print(len(out))
    return {"spectrum": torch.tensor(out)}


# for training we drop chunks of the spectrum
def drop_chunks(batch, size=15):
    batch["input"] = batch["spectrum"].clone()
    # random start location between 0 and length of the spectrum
    start = torch.randint(0, batch["spectrum"].shape[1] - size, (1,)).item()
    batch["input"][:, start : start + size] *= 0
    return batch


from dataclasses import dataclass


@dataclass
class SpecConfig:
    input_dim: int
    embed_dim: int
    num_layers: int
    num_heads: int
    max_len: int
    dropout: float = 0.1
    norm_first: bool = False


import torch.nn as nn
import torch.nn.functional as F


class SpecFormer(nn.Module):
    config: SpecConfig

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed = nn.Linear(config.input_dim, config.embed_dim)

        self.head = nn.Linear(config.embed_dim, 1)

        self.abs_pos = nn.Embedding(config.max_len, config.embed_dim)

        trans_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.embed_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=config.norm_first,
        )

        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=trans_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, input):
        t = input.shape[1]

        if len(input.shape) == 2:
            input = input.unsqueeze(-1)

        x = self.embed(input) + self.abs_pos.weight[:t].unsqueeze(0)

        x = F.gelu(self.encoder_stack(x))

        # adding the input back in so we model the difference
        x = input + self.head(x)

        return x


my_config = SpecConfig(
    input_dim=196,
    embed_dim=256,
    num_layers=4,
    num_heads=4,
    max_len=80,
    dropout=0.1,
    norm_first=False,
)

my_model = SpecFormer(my_config)
import torch.optim as optim

# Define the loss function
loss_fn = nn.MSELoss()


def train_model(model, loader, num_epochs, lr, weight_decay):

    model.cuda()

    # Define the optimizer
    optimizer = optim.AdamW(my_model.parameters(), lr=lr, weight_decay=weight_decay)

    try:
        # Training loop
        for epoch in range(num_epochs):
            # Set the model to train mode
            model.train()

            # Initialize the total loss for this epoch
            total_loss = 0

            i = 0

            # Iterate over the training data
            for batch in loader:

                if i % 100 == 0:
                    print(f"Batch {i} of {len(loader)}")

                # preprocess the batch
                batch = drop_chunks(preprocess(batch))

                # Clear the gradients
                optimizer.zero_grad()

                # add cosine annealing scheduler
                sched = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, len(loader), lr / 100
                )

                # Forward pass
                output = model(batch["input"].cuda())

                # Compute the loss
                loss = loss_fn(output, batch["spectrum"].cuda())

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                sched.step()

                # Update the total loss
                total_loss += loss.item()

            # Compute the average loss for this epoch
            avg_loss = total_loss / len(loader["train"])

            # Print the average loss for this epoch
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


train_model(my_model, loader["train"], num_epochs=10, lr=1e-4, weight_decay=1e-3)
torch.save(
    {"model_state_dict": my_model.state_dict(), "config": my_config},
    "trained_model.pth",
)
