from typing import Union

from lightning import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf


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
