"""
I would like to have this package as typhon.collocations. I have to talk with Gerrit about it since his typhon.datasets 
classes already provide advanced collocation methods. Maybe this should only be used as wrapper?
"""

from .dataset import * # noga

__all__ = [s for s in dir() if not s.startswith('_')]