import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWithWarmupLR(LRScheduler):
    """A cosine-annealing learning rate scheduler with initial warmup.

    Currently this cuts off after one cycle. The interface is otherwise compatible with
    :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.

    :param optimizer: wrapped optimizer
    :param T_max: maximum number of iterations; unlike
        :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`, this scheduler fixes the
        learning rate to :attr:`eta_min` after :attr:`T_max` iterations
    :param T_warmup: number of steps during which to use linear warmup
    :param eta_min: minimum learning rate
    :param last_epoch: index of last epoch
    :param verbose: whether to print a message to `stdout` for each update
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        T_warmup: int = 0,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            print(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        if self.last_epoch < self.T_warmup:
            # linear warmup
            # T_warmup > last_epoch >= 0 so no division by zero
            return [
                base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs
            ]
        elif self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        else:
            i = self.last_epoch - self.T_warmup
            n = self.T_max - self.T_warmup
            decay_ratio = i / n

            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

            # coeff is between 0 and 1 so lr is between eta_min and base_lr
            return [
                self.eta_min + coeff * (base_lr - self.eta_min)
                for base_lr in self.base_lrs
            ]
