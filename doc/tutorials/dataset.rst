How to use typhon.spareice.Dataset?
###################################

.. contents:: :local:

.. highlight:: python
   :linenothreshold: 5

What is the idea?
=================

Imagine you have a big dataset consisting of many files containing observations
(e.g. images or satellite data). Each file covers a certain time period and
is located in folders which names contain information about the time period.
See figure :numref:`fig-example-directory` for an example.

.. _fig-example-directory:

.. figure:: _figures/dataset_directory.png
   :alt: screen shot of dataset directory structure

   An example: All files of *Instrument A* are located in subdirectories which
   contain temporal information in their names (year, month, day, etc.).

Typical tasks to analyze this dataset would include iterating over those
files, reading them, applying functions on their content and eventually
adding files with new data to this dataset. So, how to find all files in a
time period? You could start by writing nested *for* loops and using
python's *glob* function. Normally, such solutions requires time to
implement, are error-prone and are not portable to other datasets with
different structures. Hence, save your time/energy/nerves and simply use
the :class:`~typhon.spareice.datasets.Dataset` class.

Quick Start
===========

We stick to our example from above and want to find all files from our
*Instrument A* dataset between two dates. To do this, we need to initialize a
Dataset object and tell it where to find our files:

.. code-block:: python

   # Import the Dataset class from the typhon module.
   from typhon.spareice.datasets import Dataset

   # Define a dataset object with the files.
   instrument_A = Dataset(
      "Data/InstrumentA/{year}/{month}/{day}/data_{hour}-{minute}-{second}.nc"
   )

What happens in this piece of code? We import the Dataset class from the typhon
module and define the Dataset object, give it a name and tell it where
to find its files. We do the latter by giving the generalized path
pattern pointing to each file instead of giving explicit paths. The words
surrounded by braces (e.g. "{year}") are called placeholders. They define
what information can be retrieved from the filename. If you want to know
more about those placeholders, have look at the section
:ref:`sec-placeholders`.

We want to print the names and time coverages of all files from the 1st of
January 2016 (the whole day, i.e. from 0-24h).

.. code-block:: python

    # Find all files between 01/01/2016 and 02/01/2016:
    for file in instrument_A.find_files("2016-01-01", "2016-01-02"):
        print(file)

.. code-block:: none
   :caption: Output:

   File: Data/InstrumentA/2016/01/01/data_00-00-00.nc
       Start: 2016-01-01 00:00:00
       End: 2016-01-01 00:00:00
   File: Data/InstrumentA/2016/01/01/data_06-00-00.nc
       Start: 2016-01-01 06:00:00
       End: 2016-01-01 06:00:00
   File: Data/InstrumentA/2016/01/01/data_12-00-00.nc
       Start: 2016-01-01 12:00:00
       End: 2016-01-01 12:00:00
   File: Data/InstrumentA/2016/01/01/data_18-00-00.nc
       Start: 2016-01-01 18:00:00
       End: 2016-01-01 18:00:00

The :meth:`~typhon.spareice.datasets.Dataset.find_files` method find all
files between two dates and returns their names and time coverages (start
and end times). If we want to sort them by their starting times, we can set
its *sort* parameter to true.

Read and Create Files
=====================

The Dataset class has more interesting functionality that we are going to
investigate in more detail later. But before doing this, we have to understand
how we can open and read files from one dataset. Since there are a lot of
different types of datasets out there and each one of them might have its own
file format, the Dataset object needs help from you in order to
handle those files. You must tell the Dataset how to read and write its
files by giving a *file handler* to it. A file handler is an object that
can read a file in a certain format or write data to it. For example, if we
want to read the files from our *Instrument A* and print out their content, we
need a file handler that can handle those files. The files are stored in the
NetCDF4 format. Lucky for us, there is a file handler class that can handle
such files (:class:`~typhon.spareice.handlers.commom.NetCDF4`, for a complete
list of official handler classes in typhon have a look at
:ref:`typhon-handlers`). The only thing that we need to do now, is giving this
file handler object to the dataset object during initialization:

.. code-block:: python

   # Import the Dataset class from the typhon module.
   from typhon.spareice.datasets import Dataset
   from typhon.spareice.handlers.common import NetCDF4

   # Define a dataset object with the files.
   instrument_A = Dataset(
      "Data/InstrumentA/{year}/{month}/{day}/data_{hour}-{minute}-{second}.nc",
      # With the next line, the dataset object knows how to handle its files:
      handler=NetCDF4(),
   )

The dataset object knows how to open our files now. We can try it by using the
:meth:`~typhon.spareice.datasets.Dataset.read` method:

.. code-block:: python

   # Open all files between 01/01/2016 and 02/01/2016:
   for file in instrument_A.find_files("2016-01-01", "2016-01-02"):
      print(file)
      data = instrument_A.read(file)
      print(data)

.. code-block:: none
   :caption: Output:

   File: ../../Data/InstrumentA/2016/01/01/data_00-00-00.nc
       Start: 2016-01-01 00:00:00, End: 2016-01-01 00:00:00
   <xarray.Dataset>
   Dimensions:  (dim_0: 100)
   Dimensions without coordinates: dim_0
   Data variables:
       x        (dim_0) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16  ...
       y        (dim_0) float64 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 ...
   File: ../../Data/InstrumentA/2016/01/01/data_06-00-00.nc
       Start: 2016-01-01 06:00:00, End: 2016-01-01 06:00:00
   ...

How does this work? All file handler objects (i.e.
:class:`~typhon.spareice.handlers.commom.NetCDF4` as well) have a *read* method
implemented. When we call
:meth:`~typhon.spareice.datasets.Dataset.read`, the dataset object simply calls
the :meth:`~typhon.spareice.handlers.commom.NetCDF4.read` method and redirects
its output to us. The same works with creating files, when the file handler
object has implemented a *write* method.

These are the special methods that are used by
:class:`~typhon.spareice.datasets.Dataset`:

+---------------------+-----------------------+-------------------------------+
| Dataset method      | FileHandler method    | Description                   |
+=====================+=======================+===============================+
| Dataset.read()      | FileHandler.read()    | Opens and reads a file.       |
+---------------------+-----------------------+-------------------------------+
| Dataset.write()     | FileHandler.write()   | Writes data to a file.        |
+---------------------+-----------------------+-------------------------------+
| Dataset.get_info()  | FileHandler.get_info()| Gets information (e.g. time \ |
|                     |                       | coverage) of a file.          |
+---------------------+-----------------------+-------------------------------+

We could use both methods to change the content of each file:

.. code-block:: python

   for file in instrument_A.find_files("2016-01-01", "2016-01-02"):
       # Open file:
       data = instrument_A.read(file)

       # Change content:
       data["x"] /= 2

       # Overwrite the old file:
       instrument_A.write(file, data)



**TODO: Finish tutorial**

Get information about the file
==============================


.. _typhon-dataset-placeholders:

Placeholders
============

Further recipes
===============


Find all files in a period
--------------------------




Read all files in a period
--------------------------


Use multiple processes
----------------------


Use magic indexing
------------------


Find overlapping files between two datasets
-------------------------------------------
