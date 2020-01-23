import functools
import logging
from os.path import dirname, join

from . import arts  # noqa
from . import cloudmask  # noqa
from . import config  # noqa
from . import constants  # noqa
from . import files  # noqa
from . import geodesy  # noqa
from . import geographical  # noqa
from . import latex  # noqa
from . import math  # noqa
from . import nonlte  # noqa
from . import physics  # noqa
from . import plots  # noqa
from . import spectroscopy  # noqa
from . import topography  # noqa
from . import trees  # noqa
from . import utils  # noqa
from .environment import environ  # noqa

# Parse version number from module-level ASCII file
__version__ = open(join(dirname(__file__), "VERSION")).read().strip()


def test():
    """Use pytest to collect and run all tests in typhon.tests."""
    import pytest

    return pytest.main(["--pyargs", "typhon.tests"])


_logger = logging.getLogger(__name__)


@functools.lru_cache()
def _ensure_handler(handler=None, formatter=None):
    """Make sure that a handler is attached to the root logger.

    The LRU cache ensures that a new handler is only created during the
    first call of the function. From then on, this handler is reused.
    """
    if handler is None:
        handler = logging.StreamHandler()

    if formatter is None:
        formatter = logging.Formatter(logging.BASIC_FORMAT)

    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    return handler


def set_loglevel(level, handler=None, formatter=None):
    """Set the loglevel of the package.

    Parameters:
        level (int): Loglevel according to the ``logging`` module.
        handler (``logging.Handler``): Logging handler.
        formatter (``logging.Formatter``): Logging formatter.
    """
    _logger.setLevel(level)
    _ensure_handler(handler, formatter).setLevel(level)


def set_fancy_logging(level=None):
    """Create a basic logging config with colorful output format."""
    color = "\033[1;%dm"
    reset = "\033[0m"
    black, red, green, yellow, blue, magenta, cyan, white = [
        color % (30 + i) for i in range(8)
    ]
    logformat = (
        "["
        f"{magenta}%(levelname)s{reset}:"
        f"{red}%(asctime)s.%(msecs)03d{reset}:"
        f"{yellow}%(filename)s{reset}"
        f":{blue}%(lineno)s{reset}"
        f":{green}%(funcName)s{reset}"
        f"] %(message)s"
    )

    logging.basicConfig(
        format=logformat,
        level=level if level is not None else logging.INFO,
        datefmt="%H:%M:%S",
    )
