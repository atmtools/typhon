"""Utilities related to caching and memoisation
"""

import functools
import logging

def mutable_cache(maxsize=10):
    """In-memory cache like functools.lru_cache but for any object

    This is a re-implementation of functools.lru_cache.  Unlike
    functools.lru_cache, it works for any objects, mutable or not.  That
    probably means it is sometimes wrong.  Please use with caution and
    report any bugs to the typhon issue tracker.

    If you call the *resulting* function with a keyword argument
    'CLEAR_CACHE', the cache will be cleared.  Otherwise, cache is rotated
    when more than `maxsize` elements exist in the cache.  Be careful with
    functions returning large objects, such as reading routines for
    datasets like IASI.  Everything is kept in RAM!

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
        keylist = [] # don't make it too long

        def wrapper(*args, **kwds):
            # Problem with pickle: dataset objects (commonly passed as
            # 'self') contain a cache which is a shelve object which
            # cannot be pickled.  Would need to create a proper pickle
            # protocol for dataset objects... maybe later
            #key = pickle.dumps(args, 1) + pickle.dumps(kwds, 1)
            key = str(args) + str(kwds)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                logging.debug(("Getting result from cache "
                    " (key {!s}").format(key))
                return result
            if kwds.get("CLEAR_CACHE"):
                del kwds["CLEAR_CACHE"]
                cache.clear()
                keylist.clear()
#            logging.debug("No result in cache")
            result = user_function(*args, **kwds)
#            logging.debug("Storing result in cache")
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

