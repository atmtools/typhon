"""Utilities related to caching and memoisation
"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

import functools
import logging
import copy


__all__ = [
    'mutable_cache',
]


logger = logging.getLogger(__name__)
def mutable_cache(maxsize=10):
    """In-memory cache like functools.lru_cache but for any object

    This is a re-implementation of functools.lru_cache.  Unlike
    functools.lru_cache, it works for any objects, mutable or not.
    Therefore, it returns returns a copy and it is wrong if the mutable
    object has changed!  Use with caution!

    If you call the *resulting* function with a keyword argument
    'CLEAR_CACHE', the cache will be cleared.  Otherwise, cache is rotated
    when more than `maxsize` elements exist in the cache.  Additionally,
    if you call the resulting function with NO_CACHE=True, it doesn't
    cache at all.  Be careful with functions returning large objects, such
    as reading routines for datasets like IASI.  Everything is kept in RAM!

    Args:
        maxsize (int): Maximum number of return values to be remembered.

    Returns:
        New function that has caching implemented.
    """

    sentinel = object()
    make_key = functools._make_key

    def decorating_function(user_function):
        cache = {}
        cache_get = cache.get
        keylist = []  # don't make it too long

        def wrapper(*args, **kwds):
            if kwds.get("CLEAR_CACHE"):
                del kwds["CLEAR_CACHE"]
                cache.clear()
                keylist.clear()
            if kwds.get("NO_CACHE"):
                del kwds["NO_CACHE"]
                return user_function(*args, **kwds)
            elif "NO_CACHE" in kwds:
                del kwds["NO_CACHE"]
            # Problem with pickle: dataset objects (commonly passed as
            # 'self') contain a cache which is a shelve object which
            # cannot be pickled.  Would need to create a proper pickle
            # protocol for dataset objects... maybe later
            # key = pickle.dumps(args, 1) + pickle.dumps(kwds, 1)
            key = str(args) + str(kwds)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                logger.debug(("Getting result for {!s} from cache "
                               " (key {!s}").format(user_function, key))
                # make sure we return a copy of the result; when a = f();
                # b = f(), users should reasonably expect that a is not b.
                return copy.copy(result)
#            logger.debug("No result in cache")
            result = user_function(*args, **kwds)
#            logger.debug("Storing result in cache")
            cache[key] = result
            keylist.append(key)
            if len(keylist) > maxsize:
                try:
                    del cache[keylist[0]]
                    del keylist[0]
                except KeyError:
                    pass
            return result

        return functools.update_wrapper(wrapper, user_function)

    return decorating_function
