#!/usr/bin/env python

"""Contains metaclasses for typhon development and others

This module contains metaclasses that are used within typhon, but that may
also be useful for others.
"""

import abc

import numpy

# Use a metaclass to inherit docstrings
#
# http://stackoverflow.com/a/8101118/974555
class DocStringInheritor(type):
    """Automatically inherit docstrings.
    
    If overriding a method for a class with a DocStringInheritor
    metaclass, the subclass can omit the docstring.  The docstring will be
    inherited from the superclass.
    
    This is inspired by the default behaviour in MatlabⓇ.  The
    implementation is inspired by Paul Mcguire, see 
    http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
    """
    def __new__(meta, name, bases, clsdict):
        if not('__doc__' in clsdict and clsdict['__doc__']):
            for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
                doc=mro_cls.__doc__
                if doc:
                    clsdict['__doc__']=doc
                    break
        for attr, attribute in clsdict.items():
            if not attribute.__doc__:
                for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()
                                if hasattr(mro_cls, attr)):
                    doc=getattr(getattr(mro_cls,attr),'__doc__')
                    if doc:
                        attribute.__doc__=doc
                        # added by Gerrit
                        attribute.__doc__ += ("\n\nDocstring inherited "
                            "from {:s}".format(mro_cls.__name__))
                        break
        # GH: replaced `type` by super() to not break multiple inheritance
        return super().__new__(meta, name, bases, clsdict)

class AbstractDocStringInheritor(DocStringInheritor, abc.ABCMeta):
    """Automatically inherit docstrings for abstract classes.

    For details, see DocStringInheritor.
    """
    pass
