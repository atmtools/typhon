import functools
import logging

from .version import __version__

try:
    __TYPHON_SETUP__
except:
    __TYPHON_SETUP__ = False

if not __TYPHON_SETUP__:
    from . import arts
    from . import cloudmask
    from . import config
    from . import constants
    from . import files
    from . import geodesy
    from . import geographical
    from . import latex
    from . import math
    from . import nonlte
    from . import physics
    from . import plots
    from . import spectroscopy
    from . import trees
    from . import utils
    from .environment import environ


    def test():
        """Use pytest to collect and run all tests in typhon.tests."""
        import pytest

        return pytest.main(['--pyargs', 'typhon.tests'])


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
        color % (30 + i) for i in range(8)]
    logformat = (
        '['
        f'{magenta}%(levelname)s{reset}:'
        f'{red}%(asctime)s.%(msecs)03d{reset}:'
        f'{yellow}%(filename)s{reset}'
        f':{blue}%(lineno)s{reset}'
        f':{green}%(funcName)s{reset}'
        f'] %(message)s'
    )

    logging.basicConfig(
        format=logformat,
        level=level if level is not None else logging.INFO,
        datefmt = '%H:%M:%S',
    )
