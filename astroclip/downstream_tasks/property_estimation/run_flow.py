
import h5py
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import seaborn as sns
import torch
import tqdm
from astropy.table import Table, join
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})

# Overall Definitions:
properties = ["Z_HP", "LOG_MSTAR", "Z_MW", "t_ageMW", "SFR"]
property_titles = ["$Z_{HP}$", "$\log_{M_\star}$", "$Z_{MW}$", "$t_{age}$", "$SFR$"]
scaler = StandardScaler()

# Data Files
embedding_file = "/mnt/ceph/users/lparker/good_embeddings/newest_embeddings.h5py"
stein_embedding_file = "/mnt/home/lparker/ceph/stein_propertyembeddings.h5"
DINO_embedding_file = "/mnt/home/lparker/ceph/DINO_embeddings.h5"
GalFormer_embedding_file = "/mnt/home/lparker/ceph/GalFormer_embeddings.h5"


# ----- Replace with new Dataset Loader ----- #
def get_embeddings(embedding_file, source="images"):
    CLIP_embeddings = h5py.File(embedding_file, "r")
    train_embeddings = CLIP_embeddings["train"]
    test_embeddings = CLIP_embeddings["test"]

    if source == "images":
        train_table = Table(
            {
                "targetid": train_embeddings["targetid"],
                "image_features": train_embeddings["image_features"],
            }
        )

        test_table = Table(
            {
                "targetid": test_embeddings["targetid"],
                "image_features": test_embeddings["image_features"],
            }
        )

    elif source == "spectra":
        train_table = Table(
            {
                "targetid": train_embeddings["targetid"],
                "spectra_features": train_embeddings["spectra_features"],
            }
        )

        test_table = Table(
            {
                "targetid": test_embeddings["targetid"],
                "spectra_features": test_embeddings["spectra_features"],
            }
        )

    elif source == "photometry":
        train_table = Table({"targetid": train_embeddings["targetid"]})
        test_table = Table({"targetid": test_embeddings["targetid"]})

    return train_table, test_table


def get_provabgs(embedding_file, source="images"):
    provabgs = Table.read("/mnt/home/lparker/ceph/BGS_ANY_full.provabgs.sv3.v0.hdf5")
    provabgs = provabgs[
        (provabgs["LOG_MSTAR"] > 0)
        * (provabgs["MAG_G"] > 0)
        * (provabgs["MAG_R"] > 0)
        * (provabgs["MAG_Z"] > 0)
    ]
    inds = np.random.permutation(len(provabgs))
    provabgs = provabgs[inds]

    if source == "images":
        train_table, test_table = get_embeddings(embedding_file, source)
        train_provabgs = join(
            provabgs, train_table, keys_left="TARGETID", keys_right="targetid"
        )
        test_provabgs = join(
            provabgs, test_table, keys_left="TARGETID", keys_right="targetid"
        )

    elif source == "spectra":
        train_table, test_table = get_embeddings(embedding_file, source)
        train_provabgs = join(
            provabgs, train_table, keys_left="TARGETID", keys_right="targetid"
        )
        test_provabgs = join(
            provabgs, test_table, keys_left="TARGETID", keys_right="targetid"
        )

    elif source == "photometry":
        train_table, test_table = get_embeddings(embedding_file, source)
        train_provabgs = join(
            provabgs, train_table, keys_left="TARGETID", keys_right="targetid"
        )
        test_provabgs = join(
            provabgs, test_table, keys_left="TARGETID", keys_right="targetid"
        )

    return train_provabgs, test_provabgs


# -------------------------------------------- #


def get_data(embedding_file, source="images"):
    train_provabgs, test_provabgs = get_provabgs(embedding_file, source)

    # Scale the galaxy property data
    prop_scalers = {}
    y_train, y_test = np.zeros((len(train_provabgs), 5)), np.zeros(
        (len(test_provabgs), 5)
    )
    for i, p in enumerate(properties):
        prop_train, prop_test = train_provabgs[p].reshape(-1, 1), test_provabgs[
            p
        ].reshape(-1, 1)
        if p == "Z_MW":
            prop_train, prop_test = np.log(prop_train), np.log(prop_test)
        if p == "SFR":
            prop_train, prop_test = np.log(prop_train) - train_provabgs[
                "LOG_MSTAR"
            ].reshape(-1, 1), np.log(prop_test) - test_provabgs["LOG_MSTAR"].reshape(
                -1, 1
            )
        prop_scaler = StandardScaler().fit(prop_train)
        prop_train, prop_test = prop_scaler.transform(
            prop_train
        ), prop_scaler.transform(prop_test)
        y_train[:, i], y_test[:, i] = prop_train.squeeze(), prop_test.squeeze()
        prop_scalers[p] = prop_scaler

    if source == "images":
        train_images, test_images = (
            train_provabgs["image_features"],
            test_provabgs["image_features"],
        )
        image_scaler = StandardScaler().fit(train_images)
        train_images, test_images = image_scaler.transform(
            train_images
        ), image_scaler.transform(test_images)

        data = {
            "X_train": torch.tensor(train_images),
            "X_test": torch.tensor(test_images),
            "y_train": torch.tensor(y_train, dtype=torch.float32),
            "y_test": torch.tensor(y_test, dtype=torch.float32),
        }

    elif source == "spectra":
        train_spectra, test_spectra = (
            train_provabgs["spectra_features"],
            test_provabgs["spectra_features"],
        )
        spectrum_scaler = StandardScaler().fit(train_spectra)
        train_spectra, test_spectra = spectrum_scaler.transform(
            train_spectra
        ), spectrum_scaler.transform(test_spectra)

        data = {
            "X_train": torch.tensor(train_spectra),
            "X_test": torch.tensor(test_spectra),
            "y_train": torch.tensor(y_train, dtype=torch.float32),
            "y_test": torch.tensor(y_test, dtype=torch.float32),
        }

    elif source == "photometry":
        data = {
            "X_train": torch.tensor(
                train_provabgs["MAG_G", "MAG_R", "MAG_Z"], dtype=torch.float32
            ),
            "X_test": torch.tensor(
                test_provabgs["MAG_G", "MAG_R", "MAG_Z"], dtype=torch.float32
            ),
            "y_train": torch.tensor(y_train, dtype=torch.float32),
            "y_test": torch.tensor(y_test, dtype=torch.float32),
        }

    else:
        raise ValueError("Invalid source. Must be one of: images, spectra, photometry")

    return data, prop_scalers


def get_supervised(source):
    if source == "resnet":
        resnet_results = torch.load("./baseline_models/resnet_results")
        data = {
            "X_train": torch.tensor(resnet_results["train_preds"]),
            "y_train": torch.tensor(resnet_results["train_trues"]),
            "X_test": torch.tensor(resnet_results["test_preds"]),
            "y_test": torch.tensor(resnet_results["test_trues"]),
        }
        scalers = resnet_results["scalers"]

    elif source == "spender":
        spender_results = torch.load("./baseline_models/spender_results")
        data = {
            "X_train": torch.tensor(spender_results["train_preds"]),
            "y_train": torch.tensor(spender_results["train_trues"]),
            "X_test": torch.tensor(spender_results["test_preds"]),
            "y_test": torch.tensor(spender_results["test_trues"]),
        }
        scalers = spender_results["scalers"]

    else:
        raise ValueError("Invalid source. Must be one of: resnet, spender")

    return data, scalers


class ConditionalFlowStack(dist.conditional.ConditionalComposeTransformModule):
    def __init__(self, input_dim, context_dim, hidden_dims, num_flows):
        coupling_transforms = [
            T.conditional_spline(
                input_dim,
                context_dim,
                count_bins=4,
                hidden_dims=hidden_dims,
                order="linear",
            ).cuda()
            for _ in range(num_flows)
        ]

        super().__init__(coupling_transforms, cache_size=1)


def train(
    train_loader,
    test_loader,
    val_loader,
    base_dist,
    transform,
    index,
    num_epochs=100,
    lr=5e-5,
):
    # Initialize the optimizer
    optimizer = torch.optim.Adam(transform.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    pyro.clear_param_store()

    train_losses, val_losses = [], []
    epochs = tqdm.trange(num_epochs, desc="Training flow %i" % (index))
    best_val_loss, best_transform = float("inf"), None

    for epoch in epochs:
        avg_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            flow_dist = dist.conditional.ConditionalTransformedDistribution(
                base_dist, [transform]
            ).condition(X.to("cuda"))
            train_loss = -flow_dist.log_prob(y.to("cuda")).mean()
            train_loss.backward()
            optimizer.step()
            flow_dist.clear_cache()
            avg_loss += train_loss.item()
        avg_loss /= len(train_loader)

        avg_val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                flow_dist = dist.conditional.ConditionalTransformedDistribution(
                    base_dist, [transform]
                ).condition(X.to("cuda"))
                avg_val_loss -= flow_dist.log_prob(y.to("cuda")).mean().item()
                flow_dist.clear_cache()
        avg_val_loss /= len(val_loader)

        # add early stopping
        if epoch > 5 and avg_val_loss > np.mean(val_losses[-2:]):
            break

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)

        epochs.set_description(
            "step: {}, train loss: {}, val loss: {}".format(
                epoch, np.round(avg_loss, 3), np.round(avg_val_loss, 3)
            )
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_transform = transform

        scheduler.step()

    avg_test_loss = 0
    for X, y in test_loader:
        flow_dist = dist.conditional.ConditionalTransformedDistribution(
            base_dist, [best_transform]
        ).condition(X.to("cuda"))
        avg_test_loss -= flow_dist.log_prob(y.to("cuda")).mean()
        flow_dist.clear_cache()
    avg_test_loss /= len(test_loader)

    return best_transform, avg_test_loss


def do_flow(
    data,
    num_ndes,
    n_samples=500,
    max_hidden=128,
    max_flows=4,
    min_flows=None,
    num_epochs=100,
    lr=5e-5,
    batch_size=256,
):
    if min_flows == None:
        if max_flows < 3:
            min_flows = 1
        else:
            min_flows = max_flows - 2

    X_train, X_test = data["X_train"].to("cuda"), data["X_test"].to("cuda")
    y_train, y_test = data["y_train"].to("cuda"), data["y_test"].to("cuda")

    X_train, X_val, y_train, y_val = (
        X_train[: int(0.8 * len(X_train))],
        X_train[int(0.8 * len(X_train)) :],
        y_train[: int(0.8 * len(y_train))],
        y_train[int(0.8 * len(y_train)) :],
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    input_dim, context_dim = y_train.shape[1], X_train.shape[1]

    base_dist = dist.Normal(
        torch.zeros(input_dim).to("cuda"), torch.ones(input_dim).to("cuda")
    )

    flows, losses, samples = [], [], []
    for i in range(num_ndes):
        nhidden = int(
            np.ceil(
                np.exp(np.random.uniform(np.log(max_hidden / 2), np.log(max_hidden)))
            )
        )
        nblocks = int(np.random.uniform(min_flows, max_flows))

        print("Flow %i with nhidden=%i; ncomponents=%i:" % (i, nhidden, nblocks))

        transform = ConditionalFlowStack(
            input_dim, context_dim, [nhidden, nhidden], nblocks
        ).to("cuda")
        try:
            flow, loss = train(
                train_loader,
                test_loader,
                val_loader,
                base_dist,
                transform,
                i,
                num_epochs=num_epochs,
                lr=lr,
            )
            print("Test Loss: %.3f" % (loss))

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

        except:
            print("Failed to train flow %i" % (i))
            print("")
            continue

    return flows, losses, samples


datasets = {"images": {}, "spectra": {}, "photometry": {}}

# Load the data
print("Loading Data...")
datasets["images"]["clip"], _ = get_data(embedding_file, source="images")
datasets["images"]["stein"], _ = get_data(stein_embedding_file, source="images")
datasets["images"]["dino"], _ = get_data(DINO_embedding_file, source="images")
datasets["images"]["resnet"], _ = get_supervised("resnet")

datasets["spectra"]["clip"], _ = get_data(embedding_file, source="spectra")
datasets["spectra"]["GalFormer"], _ = get_data(
    GalFormer_embedding_file, source="spectra"
)
datasets["spectra"]["spender"], _ = get_supervised("spender")

datasets["photometry"]["mlp"], _ = get_data(embedding_file, source="photometry")
print("")

# Train and Evaluate Flows
flow_results = {
    "Flows": {},
    "Samples": {},
    "NLL": {},
    "Average NLL": {},
    "Variance NLL": {},
}
print("Training and evaluating flows...\n")

hyperparams = {
    "images": {"max_hidden": 32, "max_flows": 3, "min_flows": 2, "lr": 5e-4},
    "spectra": {"max_hidden": 32, "max_flows": 3, "min_flows": 2, "lr": 5e-4},
    "photometry": {"max_hidden": 32, "max_flows": 3, "min_flows": 2, "lr": 5e-4},
}

for key in datasets.keys():
    source_datasets = datasets[key]

    print("Training and evaluating flows for %s..." % (key), flush=True)
    with open(
        f"/mnt/home/lparker/Documents/AstroFoundationModel/AstroBaselines/property_estimation/flows/new_{key}_flows_nll.txt",
        "w",
    ) as f:
        for subkey in source_datasets.keys():
            data = source_datasets[subkey]
            max_hidden, max_flows, min_flows, lr = (
                hyperparams[key]["max_hidden"],
                hyperparams[key]["max_flows"],
                hyperparams[key]["min_flows"],
                hyperparams[key]["lr"],
            )
            flow, nll, samples = do_flow(
                data,
                num_ndes=5,
                n_samples=500,
                max_hidden=max_hidden,
                max_flows=max_flows,
                min_flows=min_flows,
                num_epochs=200,
                lr=lr,
            )
            avg_nll = torch.mean(torch.stack(nll))
            var_nll = torch.var(torch.stack(nll))

            f.write(f"{subkey} Average NLL: {avg_nll} \n")
            f.write(f"{subkey} Variance NLL: {var_nll} \n")
            f.flush()

            torch.save(
                flow,
                f"/mnt/home/lparker/Documents/AstroFoundationModel/AstroBaselines/property_estimation/flows/{key}/new_{subkey}_flow.pt",
            )
            torch.save(
                samples,
                f"/mnt/home/lparker/Documents/AstroFoundationModel/AstroBaselines/property_estimation/flows/{key}/new_{subkey}_samples.pt",
            )
            print("")
