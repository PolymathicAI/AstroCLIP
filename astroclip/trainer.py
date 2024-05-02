#!/usr/bin/env python
from typing import Any, Optional

import matplotlib.pyplot as plt
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.cli import (
    ArgsType,
    LightningArgumentParser,
    LightningCLI,
    LRSchedulerTypeUnion,
)
from lightning.pytorch.loggers import WandbLogger
from torch.optim import Optimizer

from astroclip import format_with_env
from astroclip.callbacks import CustomSaveConfigCallback


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
