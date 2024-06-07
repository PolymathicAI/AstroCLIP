from argparse import Namespace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeVar
from warnings import warn

from dotenv import dotenv_values

WARN_ONCE = True


# TODO: change here the defaults
ASTROCLIP_ROOT = "/mnt/ceph/users/polymathic/astroclip"
WANDB_ENTITY_NAME = "flatiron-scipt"


def default_dotenv_values():
    """Use a default .env but tell the user how to create their own."""

    env_dir = Path(__file__).parent
    env_path = env_dir / ".env"

    if env_path.exists():
        return dotenv_values(env_path)

    with NamedTemporaryFile(mode="w+") as f:
        global WARN_ONCE

        # TODO: these should be replaced with a folder in the project's root
        f.write("ASTROCLIP_ROOT={ASTROCLIP_ROOT}\n")
        f.write('WANDB_ENTITY_NAME="{WANDB_ENTITY_NAME}"\n')
        f.flush()

        if WARN_ONCE:
            f.seek(0)
            warn(
                f"No .env file found in {env_dir}. "
                "Using default environment variables for rusty. "
                f"To suppress this warning, create {env_dir}/.env with, e.g., the following content:\n"
                f"{f.read()}"
            )
            WARN_ONCE = False

        return dotenv_values(f.name)


T = TypeVar("T")


def format_with_env(s: T) -> T:
    if isinstance(s, str):
        for k, v in default_dotenv_values().items():
            s = s.replace("{" + k + "}", v)
        return s
    elif isinstance(s, dict):
        return {k: format_with_env(v) for k, v in s.items()}
    elif isinstance(s, list):
        return [format_with_env(v) for v in s]
    elif isinstance(s, Namespace):
        return type(s)(**{k: format_with_env(v) for k, v in s.__dict__.items()})
    else:
        return s
