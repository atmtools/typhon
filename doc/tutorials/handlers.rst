Create your own file handler
############################

.. contents:: :local:

.. highlight:: python:
   :linenothreshold: 5


Introduction
============

File handlers are objects that can read or write files in a specific format.
The formats can be very common like NetCDF4 or very specified ones like the one
from SEVIRI HDF level 1.5 files. File handlers are required by the
:class:`~typhon.files.fileset.FileSet` object to load data from files or to
store data back to files. File handlers can provide up to three things:

* *Reading* of files and returning their content.
* *Writing* data to files in a specific format.
* *Getting information* about a file that cannot be retrieved from the file
    name.

File handlers for the most common and some specific instrument formats are
already included in the typhon package; have a look at :ref:`typhon-handlers`.

However, it might happen that you need to build your own file handler object.
This tutorial shows to you how one should do it.


Basis structure
===============

Your file handler needs a `read`, `write` and  a `get_info` method even if it
does not implement all of them.

+-----------------------+-------------------------------+
| FileHandler method    | Description                   |
+=======================+===============================+
| FileHandler.read()    | Opens and reads a file.       |
+-----------------------+-------------------------------+
| FileHandler.write()   | Writes data to a file.        |
+-----------------------+-------------------------------+
| FileHandler.get_info()| Gets information (e.g. time \ |
|                       | coverage) of a file.          |
+-----------------------+-------------------------------+

The best start is to inherit from the generic
:class:`~typhon.files.handlers.common.FileHandler` object:

.. code-block:: python

   # Import the file handler base class
   from typhon.files import FileHandler

   class YourFileHandler(FileHandler):
        pass
        

Reading of files
================

To provide reading, you need to implement a `read` method. It should take a
:class:`~typhon.files.handlers.common.FileInfo` object as a first argument and
return an object that holds the read content of the file (e.g. a
xarray.Dataset).

.. code-block:: python

   # Import the file handler base class
   from typhon.files import FileHandler

   class YourFileHandler(FileHandler):
        def read(self, file_info):
            """Simple reader that returns the first line of an ASCII file"""
            with open(file_info) as file:
                return file.readline()


**to be continued**