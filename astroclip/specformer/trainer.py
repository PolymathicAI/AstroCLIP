#!/usr/bin/env python
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning import LightningModule, Callback, Trainer
from lightning.pytorch.cli import (
    ArgsType,
    LightningArgumentParser,
    LightningCLI,
    LRSchedulerTypeUnion,
)
from lightning.pytorch.loggers import WandbLogger
import wandb

from torch.optim import Optimizer
from typing import Any, Optional
import matplotlib.pyplot as plt


from astroclip.specformer.utils import CustomSaveConfigCallback
from astroclip import format_with_env


class WrappedLightningCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        self.config = format_with_env(self.config)

    # Changing the lr_scheduler interval to step instead of epoch
    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule,
        optimizer: Optimizer,
        lr_scheduler: Optional[LRSchedulerTypeUnion] = None,
    ) -> Any:
        optimizer_list, lr_scheduler_list = LightningCLI.configure_optimizers(
            lightning_module, optimizer=optimizer, lr_scheduler=lr_scheduler
        )

        for idx in range(len(lr_scheduler_list)):
            if not isinstance(lr_scheduler_list[idx], dict):
                lr_scheduler_list[idx] = {
                    "scheduler": lr_scheduler_list[idx],
                    "interval": "step",
                }
        return optimizer_list, lr_scheduler_list


class PlotsCallback(Callback):

    def __init__(self) -> None:
        super().__init__()

    def plot_spectrum(self, batch, output):
        sample_id = 3

        bs = len(batch["input"])
        sp_rec = batch["target"][:, 1:, 2:99].reshape(bs, -1)[sample_id]
        in_rec = batch["input"][:, 1:, 2:99].reshape(bs, -1)[sample_id]
        out_rec = output[:, 1:, 2:99].reshape(bs, -1)[sample_id]

        # plot the moving average of the spectrum
        win = 20

        sp_rec = [sp_rec[i : i + win].mean().item() for i in range(0, len(sp_rec), win)]
        in_rec = [in_rec[i : i + win].mean().item() for i in range(0, len(in_rec), win)]
        out_rec = [
            out_rec[i : i + win].mean().item() for i in range(0, len(out_rec), win)
        ]

        fig = plt.figure()
        plt.plot(in_rec, label="dropped")
        plt.plot(sp_rec, label="original")
        plt.plot(out_rec, label="reconstructed")
        return fig

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0:
            output = trainer.model(batch["input"])
            fig = self.plot_spectrum(batch, output)
            for logger in trainer.loggers:
                # Check WandbLogger and Enabled
                if issubclass(type(logger), WandbLogger) and not issubclass(
                    type(logger.experiment), wandb.sdk.lib.disabled.RunDisabled
                ):
                    logger: WandbLogger = logger
                    logger.experiment.log({f"plot/{pl_module.current_epoch}": fig})


def main_cli(args: ArgsType = None, run: bool = True):
    cli = WrappedLightningCLI(
        save_config_kwargs={"overwrite": True},
        save_config_callback=CustomSaveConfigCallback,
        parser_kwargs={"parser_mode": "omegaconf"},
        args=args,
        run=run,
    )
    return cli


if __name__ == "__main__":
    main_cli(run=True)
