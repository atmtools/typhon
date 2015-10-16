"""A general purpose module that includes simplified pickling/unpickling
functions for saving arbitrarily complex python objects, and functions for
performing multi-threaded calculations."""

import sys
import copy

try:
    unicode = unicode
except NameError:
    unicode = str


class PyARTSError(Exception):
    pass


class PyARTSWarning(UserWarning):
    pass


def quotify(s):
    """Adds quotation marks around the string, if not present.

    This is necessary for string literals in ARTS.
    Does not take care of escaping anything.

    >>> print(quotify("Hello, world!"))
    "Hello, world!"
    """
    if s.startswith('"') and s.endswith('"'):
        return s
    else:
        return '"%s"' % s


def dict_combine_with_default(in_dict, default_dict):
    """A useful function for dealing with dictionary function input.  Combines
    parameters from in_dict with those from default_dict with the output
    having default_dict values for keys not present in in_dict"""
    out_dict = copy.deepcopy(in_dict)
    for key in default_dict.keys():
        out_dict[key] = copy.deepcopy(in_dict.get(key, default_dict[key]))
    return out_dict


def convert_to_bytes(s):
    if isinstance(s, unicode):
        return s.encode('UTF-8')
    else:
        return s


def convert_to_string(s):
    if isinstance(s, (bytes)):
        return s.decode('UTF-8')
    else:
        return s


def force_encoded_string_output(func):
    """ Decorator for __repr__
    """

    if sys.version_info.major < 3:

        def _func(*args, **kwargs):
            return func(*args, **kwargs).encode(sys.stdout.encoding or 'utf-8')

        return _func

    else:
        return func
