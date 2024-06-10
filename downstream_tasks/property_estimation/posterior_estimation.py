import os, sys
sys.path.append('../..')

import argparse as argparse
import lightning as L
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
import tqdm as tqdm
from astropy.table import Table
from baselines.data import SupervisedDataModule
from baselines.modules import SupervisedModel
from property_utils.models import ConditionalFlowStack
from torch.utils.data import DataLoader, TensorDataset, random_split

from astroclip.env import format_with_env

ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")

# Get the property vector that we are looking at
PROPERTIES = ["LOG_MSTAR", "Z_MW", "TAGE_MW", "sSFR"]

# Let's keep track of all of the models we have
IMAGE_MODELS = ["astroclip_image", "astrodino", "stein"]
SPECTRUM_MODELS = ["astroclip_spectrum", "specformer"]
SUPERVISED_IMAGE_MODELS = ["resnet18"]
SUPERVISED_SPECTRUM_MODELS = ["supervised_specformer", "conv+att"]


def _get_properties(train_dataset, test_dataset, properties=PROPERTIES):
    # Set up the property data
    train_property_data, test_property_data, scale = {}, {}, {}
    for p in properties:
        train_prop = torch.tensor(train_dataset[p].data, dtype=torch.float32)
        test_prop = torch.tensor(test_dataset[p].data, dtype=torch.float32)

        mean, std = train_prop.mean(), train_prop.std()
        train_property_data[p] = ((train_prop - mean) / std).squeeze()
        test_property_data[p] = ((test_prop - mean) / std).squeeze()
        scale[p] = {"mean": mean.numpy(), "std": std.numpy()}

    y_train = torch.stack([train_property_data[p] for p in properties], dim=1)
    y_test = torch.stack([test_property_data[p] for p in properties], dim=1)
    return y_train, y_test, scale


def _get_data(train_path, test_path, source):
    # Load the data
    train_provabgs = Table.read(train_path)
    test_provabgs = Table.read(test_path)

    if source in IMAGE_MODELS or source in SPECTRUM_MODELS:
        # Get embeddings
        X_train, X_test = torch.tensor(
            train_provabgs[f"{source}_embeddings"].data
        ), torch.tensor(test_provabgs[f"{source}_embeddings"].data)
        mean, std = X_train.mean(), X_train.std()
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    elif source in SUPERVISED_IMAGE_MODELS or source in SUPERVISED_SPECTRUM_MODELS:
        # Get modality
        modality = "image" if source in SUPERVISED_IMAGE_MODELS else "spectrum"

        # Set up the data module
        X_train = torch.tensor(train_provabgs[modality].data)
        X_test = torch.tensor(test_provabgs[modality].data)
        train_loader = DataLoader(
            TensorDataset(X_train), batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test), batch_size=batch_size, shuffle=False
        )

        # Load the model
        if source == "supervised_specformer":
            source = "specformer"
        state_dict = torch.load(
            f"{ASTROCLIP_ROOT}/supervised/{modality}/{source}/global_properties/model.pt"
        )
        model = SupervisedModel(
            source,
            modality=modality,
            properties=PROPERTIES,
            scale=None,
            num_epochs=None,
        )
        model.load_state_dict(state_dict)
        model.eval(), model.to("cuda")

        # Get the embeddings
        X_train, X_test = [], []

        with torch.no_grad():
            for X in tqdm.tqdm(train_loader, desc="Extracting embeddings from train"):
                X_train.append(model(X[0].to("cuda")).detach().cpu())
            for X in tqdm.tqdm(test_loader, desc="Extracting embeddings from test"):
                X_test.append(model(X[0].to("cuda")).detach().cpu())

        X_train = torch.concatenate(X_train, dim=0)
        X_test = torch.concatenate(X_test, dim=0)

    elif source == "photometry":
        X_train = torch.tensor(
            np.stack(
                [
                    train_provabgs["MAG_G"],
                    train_provabgs["MAG_R"],
                    train_provabgs["MAG_Z"],
                ]
            ),
            dtype=torch.float32,
        ).permute(1, 0)
        X_test = torch.tensor(
            np.stack(
                [
                    test_provabgs["MAG_G"],
                    test_provabgs["MAG_R"],
                    test_provabgs["MAG_Z"],
                ]
            ),
            dtype=torch.float32,
        ).permute(1, 0)

        # Normalize photometry
        mean, std = X_train.mean(), X_train.std()
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    else:
        raise ValueError("Source not found")

    # Get the scaled property values
    y_train, y_test, scale = _get_properties(train_provabgs, test_provabgs)
    return X_train, X_test, y_train, y_test, scale


def train_flow(
    train_loader,
    val_loader,
    test_loader,
    base_dist,
    transform,
    index,
    num_epochs=100,
    lr=5e-5,
):
    # Initialize the optimizer and scheduler
    optimizer = torch.optim.Adam(transform.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Clear the param store
    pyro.clear_param_store()

    # Train the flow
    train_losses, val_losses = [], []
    epochs = tqdm.trange(num_epochs, desc="Training flow %i" % (index))
    best_val_loss, best_transform = float("inf"), None
    for epoch in epochs:
        avg_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()

            # Condition the flow on the input
            flow_dist = dist.conditional.ConditionalTransformedDistribution(
                base_dist, [transform]
            ).condition(X.to("cuda"))

            # Compute the loss
            train_loss = -flow_dist.log_prob(y.to("cuda")).mean()
            train_loss.backward()
            optimizer.step()
            flow_dist.clear_cache()
            avg_loss += train_loss.item()
        avg_loss /= len(train_loader)

        avg_val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                # Condition the flow on the input
                flow_dist = dist.conditional.ConditionalTransformedDistribution(
                    base_dist, [transform]
                ).condition(X.to("cuda"))

                # Compute the loss
                avg_val_loss -= flow_dist.log_prob(y.to("cuda")).mean().item()
                flow_dist.clear_cache()
        avg_val_loss /= len(test_loader)

        # add early stopping
        if epoch > 10 and avg_val_loss > np.mean(val_losses[-5:]):
            break

        # Store the losses
        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)

        # Update the progress bar
        epochs.set_description(
            "step: {}, train loss: {}, val loss: {}".format(
                epoch, np.round(avg_loss, 3), np.round(avg_val_loss, 3)
            )
        )

        # Update the best transform
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_transform = transform

        # Step the scheduler
        scheduler.step()

    # Compute the test loss
    avg_test_loss = 0
    for X, y in test_loader:
        flow_dist = dist.conditional.ConditionalTransformedDistribution(
            base_dist, [best_transform]
        ).condition(X.to("cuda"))
        avg_test_loss -= flow_dist.log_prob(y.to("cuda")).mean().item()
        flow_dist.clear_cache()
    avg_test_loss /= len(test_loader)
    return best_transform, avg_test_loss


def run_flow_for_model(
    source: str,
    train_path: str,
    test_path: str,
    num_ndes: int = 10,
    max_hidden: int = 32,
    max_flows: int = 4,
    min_flows: int = 2,
    num_epochs: int = 25,
    lr: float = 5e-4,
    batch_size: int = 128,
    sample: bool = False,
    n_samples: int = 100,
):
    # Get data
    X_train, X_test, y_train, y_test, scale = _get_data(train_path, test_path, source)

    # Get validation dataset
    train_dataset = TensorDataset(X_train, y_train)
    train_dataset, val_dataset = random_split(
        train_dataset,
        [
            int(0.8 * len(train_dataset)),
            len(train_dataset) - int(0.8 * len(train_dataset)),
        ],
    )
    test_dataset = TensorDataset(X_test, y_test)

    # Set up loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set up the base distribution
    input_dim, context_dim = y_train.shape[1], X_train.shape[1]
    base_dist = dist.Normal(
        torch.zeros(input_dim).to("cuda"), torch.ones(input_dim).to("cuda")
    )

    # Train the ensemble of flows
    flows, losses, samples = [], [], []
    for i in range(num_ndes):
        # Sample from parameter sizes
        nhidden = int(
            np.ceil(
                np.exp(np.random.uniform(np.log(max_hidden / 2), np.log(max_hidden)))
            )
        )
        nblocks = int(np.random.uniform(min_flows, max_flows))

        print("Flow %i with nhidden=%i, ncomponents=%i..." % (i, nhidden, nblocks))

        # Set up the transform
        transform = ConditionalFlowStack(
            input_dim, context_dim, [nhidden, nhidden, nhidden], nblocks
        ).to("cuda")

        # Train the flow
        flow, loss = train_flow(
            train_loader,
            val_loader,
            test_loader,
            base_dist,
            transform,
            i,
            num_epochs=num_epochs,
            lr=lr,
        )
        print("Test Loss: %.3f" % (loss))

        # Sample from the flow
        if sample:
            flow_samples = torch.zeros((X_test.shape[0], n_samples, 5))
            with torch.no_grad():
                for i, x in enumerate(X_test):
                    flow_dist = dist.conditional.ConditionalTransformedDistribution(
                        base_dist, [flow]
                    ).condition(x)
                    flow_samples[i, :, :] = (
                        flow_dist.sample(
                            torch.Size(
                                [
                                    n_samples,
                                ]
                            )
                        )
                        .detach()
                        .cpu()
                    )
            samples.append(flow_samples)

        flows.append(flow)
        losses.append(loss)
        print("")

    # Get the metrics
    avg_nll = torch.mean(torch.tensor(losses))
    var_nll = torch.std(torch.tensor(losses))
    print("Average NLL: %.3f" % (avg_nll), "Variance NLL: %.3f" % (var_nll))
    return flows, losses, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a flow for a given model")
    parser.add_argument("--source", type=str, help="The source model to use")
    parser.add_argument("--train_path", type=str, help="The path to the training data")
    parser.add_argument("--test_path", type=str, help="The path to the testing data")
    parser.add_argument(
        "--num_ndes", type=int, default=10, help="The number of flows to train"
    )
    parser.add_argument(
        "--max_hidden", type=int, default=32, help="The maximum number of hidden units"
    )
    parser.add_argument(
        "--max_flows", type=int, default=4, help="The maximum number of flows"
    )
    parser.add_argument(
        "--min_flows", type=int, default=2, help="The minimum number of flows"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=25, help="The number of epochs to train for"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="The learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size")
    parser.add_argument(
        "--sample", type=bool, default=False, help="Whether to sample from the flow"
    )
    parser.add_argument(
        "--n_samples", type=int, default=100, help="The number of samples to draw"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="The path to save the results"
    )

    args = parser.parse_args()
    flows, losses, samples = run_flow_for_model(
        args.source,
        args.train_path,
        args.test_path,
        args.num_ndes,
        args.max_hidden,
        args.max_flows,
        args.min_flows,
        args.num_epochs,
        args.lr,
        args.batch_size,
        args.sample,
        args.n_samples,
    )

    if args.save_path is not None:
        torch.save(
            {"flows": flows, "losses": losses, "samples": samples}, args.save_path
        )
        print("Results saved to %s" % (args.save_path))
