from typing import Any, Union

import matplotlib.pyplot as plt
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf


def _safe_eval(s: str, max_len: int = 1024) -> Union[int, float]:
    """Safely evaluate an arithmetic expression.

    :param s: expression to evaluate; should only contain numbers, spaces, or the
        symbols `+, -, *, /, _, (, )`; exponential notation is supported
    :param max_len: maximum length string that will be evaluated; longer strings raise
        a `ValueError`
    """
    # XXX need to be smarter about this
    is_safe = all(ch in "e0123456789_+-*/(). " for ch in s)
    if not is_safe:
        raise ValueError(
            "Only simple arithmetic expressions involving digits, parentheses, "
            "the letter e, or the symbols '+-*/_.' are allowed"
        )
    if len(s) > max_len:
        raise ValueError(f"String length is {len(s)}, maximum allowed is {max_len}")
    return eval(s)


# allow for the ${eval:...} resolver in the config file to perform simple arithmetic
# XXX problem: accessing a node that involves resolving with `eval` is ~60x slower
#     than a simple numeric node (similar slowdowns for `oc` resolvers)
OmegaConf.register_new_resolver("eval", _safe_eval, use_cache=True)


class CustomWandbLogger(WandbLogger):
    # Disable unintended hyperparameter logging (already saved on init)
    def log_hyperparams(self, *args, **kwargs):
        ...


class CustomSaveConfigCallback(SaveConfigCallback):
    """Saves full training configuration
    Otherwise wandb won't log full configuration but only flattened module and data hyperparameters
    """

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        for logger in trainer.loggers:
            if issubclass(type(logger), WandbLogger):
                logger.experiment.config.update(self.config.as_dict())
        return super().save_config(trainer, pl_module, stage)


class PlotsCallback(Callback):
    # TODO: Update with latest code
    def __init__(self) -> None:
        super().__init__()

    def plot_spectrum(self, batch, output):
        sample_id = 3

        bs = len(batch["spectrum"])
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
        plt.plot(sp_rec, label="original")
        plt.plot(in_rec, label="dropped")
        plt.plot(out_rec, label="reconstructed")
        plt.legend()
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
                    logger.experiment.log({f"plot/{pl_module.current_epoch:03d}": fig})
