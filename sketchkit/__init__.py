__version__ = "0.0.1"

import os

import dotenv

from . import datasets
from . import core
from . import renderer
from . import utils


__all__ = ["datasets", "core", "renderer", "utils"]

HOME_ENV_PATH = os.path.expanduser("~/.sketchkit/config")
if os.path.exists(HOME_ENV_PATH):
    dotenv.load_dotenv(HOME_ENV_PATH)
else:
    os.makedirs(os.path.dirname(HOME_ENV_PATH), exist_ok=True)
    with open(HOME_ENV_PATH, "w") as f:
        f.write("")
