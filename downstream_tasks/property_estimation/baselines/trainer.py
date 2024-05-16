import os
from argparse import ArgumentParser

import lightning as L
import numpy as np
import torch
from astropy.table import Table
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader

from astroclip.env import format_with_env
from data import SupervisedDataModule
from modules import SupervisedModel

ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")


def _get_predictions(model, test_loader, test_provabgs, scale, device="cuda"):
    """Use model to get predictions"""
    test_pred = []
    with torch.no_grad():
        for X_batch in test_loader:
            y_pred = model(X_batch[0].to(device)).squeeze().detach().cpu()
            test_pred.append(y_pred)
    test_pred = torch.cat(test_pred).numpy()

    pred_dict = {}
    for i, p in enumerate(scale.keys()):
        if len(test_pred.shape) > 1:
            pred_dict[p] = (test_pred[:, i] * scale[p]["std"]) + scale[p]["mean"]
        else:
            pred_dict[p] = (test_pred * scale[p]["std"]) + scale[p]["mean"]
        print(f"{p} R^2: {r2_score(test_provabgs[p], pred_dict[p])}")

    return pred_dict


def train_baseline(
    train_dataset: str,
    test_dataset: str,
    save_path: str,
    modality: str,
    model_name: str,
    num_epochs: int = 100,
    learning_rate: float = 5e-4,
    properties: str = None,
    accelerator: str = "gpu",
):
    # Load the data
    train_provabgs = Table.read(train_dataset)
    test_provabgs = Table.read(test_dataset)

    # Define output directory avoiding collisions
    save_dir_base = os.path.join(save_path, modality, model_name, properties)
    save_dir = save_dir_base
    v_int = 0  # Suffix to add in case of collisions
    while os.path.exists(save_dir):
        print(f"Directory {save_dir} already exists, adding suffix")
        v_int += 1
        save_dir = f"{save_dir_base}-v{v_int}"

    # Define the properties to predict
    if properties == "redshift":
        property_list = ["Z_HP"]
    elif properties == "global_properties":
        property_list = ["LOG_MSTAR", "Z_MW", "TAGE_MW", "sSFR"]
    else:
        raise ValueError(
            "Invalid properties, choose from redshift or global_properties."
        )

    # Get the data loaders & normalization
    data_module = SupervisedDataModule(
        train_provabgs,
        test_provabgs,
        modality,
        properties=property_list,
    )
    data_module.setup(stage="fit")

    # Get the model
    model = SupervisedModel(
        model_name=model_name,
        modality=modality,
        properties=property_list,
        scale=data_module.scale,
        lr=learning_rate,
        num_epochs=num_epochs,
        save_dir=save_dir,
    )

    # Set up val loss checkpoint
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=save_dir,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    # Train and fit the model
    trainer = L.Trainer(
        accelerator=accelerator, max_epochs=num_epochs, callbacks=[checkpoint_callback]
    )
    trainer.fit(model, datamodule=data_module)

    # Test the model
    best_model_path = checkpoint_callback.best_model_path
    model = SupervisedModel.load_from_checkpoint(
        best_model_path,
        model_name=model_name,
        modality=modality,
        properties=property_list,
        scale=data_module.scale,
        lr=learning_rate,
        num_epochs=num_epochs,
        save_dir=save_dir,
    )

    # Get the predictions
    pred_dict = _get_predictions(
        model.model,
        data_module.test_dataloader(),
        test_provabgs,
        data_module.scale,
        device="cuda" if accelerator == "gpu" else "cpu",
    )

    # Save the model and the predictions
    print(f"Saving in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(pred_dict, os.path.join(save_dir, "test_pred.pt"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dataset",
        type=str,
        help="Path to the training dataset",
        default=f"{ASTROCLIP_ROOT}/datasets/provabgs/provabgs_paired_train.hdf5",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Path to the test dataset",
        default=f"{ASTROCLIP_ROOT}/datasets/provabgs/provabgs_paired_test.hdf5",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save the model and predictions",
        default=f"{ASTROCLIP_ROOT}/supervised/",
    )
    parser.add_argument(
        "--modality",
        type=str,
        help="Modality of the data ('image', 'spectrum', 'photometry')",
        default="image",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model to use (e.g. 'resnet18', 'astrodino', 'specformer', 'conv+att', 'mlp')",
        default="none",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train the model",
        default=50,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for the optimizer",
        default=5e-4,
    )
    parser.add_argument(
        "--properties",
        type=str,
        help="Properties to predict ('redshift' or 'global_properties')",
        default="global_properties",
    )
    args = parser.parse_args()

    # Infer model_name if missing
    if args.model_name == "none":
        if args.modality == "image":
            model_name = "resnet18"
        elif args.modality == "spectrum":
            model_name = "conv+att"
        elif args.modality == "photometry":
            model_name = "mlp"
    else:
        model_name = args.model_name

    print(
        f"Training {model_name} on {args.modality} data for {args.properties} prediction"
    )

    train_baseline(
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        save_path=args.save_dir,
        modality=args.modality,
        model_name=model_name,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        properties=args.properties,
    )
