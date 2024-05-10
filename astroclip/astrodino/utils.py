# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import dinov2.distributed as distributed
import torch
import wandb
from dinov2.eval.setup import setup_and_build_model

logger = logging.getLogger("dinov2")


class MetricLogger(object):
    def __init__(self, delimiter="\t", wandb=None, output_file=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file
        self.wandb = wandb

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not distributed.is_main_process():
            return

        os.makedirs(Path(self.output_file).parent, exist_ok=True)
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )

        metrics = {k: v.median for k, v in self.meters.items()}
        global_rank = int(os.environ.get("RANK", 0))
        if self.wandb is not None and global_rank == 0:
            self.wandb.log(metrics, step=iteration)

        dict_to_dump.update(metrics)
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")
        pass

    def log_every(
        self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0
    ):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(
                    iteration=i, iter_time=iter_time.avg, data_time=data_time.avg
                )
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / n_iterations
            )
        )


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        if not distributed.is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def setup_astrodino(
    astrodino_output_dir: str,
    astrodino_pretrained_weights: str,
    astrodino_config_file: str = "./astroclip/astrodino/config.yaml",
) -> torch.nn.Module:
    """Set up AstroDINO model"""

    # Set up config to pass to AstroDINO
    class config:
        output_dir = astrodino_output_dir
        config_file = astrodino_config_file
        pretrained_weights = astrodino_pretrained_weights
        opts = []

    sys.stdout = open(os.devnull, "w")  # Redirect stdout to null
    astrodino, _ = setup_and_build_model(config())
    sys.stderr = sys.__stderr__  # Reset stderr
    return astrodino
