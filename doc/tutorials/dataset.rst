How To Use typhon.spareice.datasets.Dataset?
============================================

.. highlight:: python
   :linenothreshold: 5

What is the idea?
-----------------

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
the :class:`typhon.spareice.datasets.Dataset` class.

Quick Start
-----------

We stick to our example from above and want to find all files from our
*Instrument A* (what a creative name!) dataset between two dates. To do this,
we need to initialize a Dataset object and tell it where to find our files:

.. code-block:: python

   # Import the Dataset class from the typhon module.
   from typhon.spareice.datasets import Dataset

   # Define a dataset object with the files.
   instrument_A = Dataset(
      name="InstrumentA",
      files="Data/InstrumentA/{year}/{month}/{day}/"
            "data_{hour}-{minute}-{second}.nc"
   )

What happen in this piece of code? We import the Dataset class from the typhon
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
   date1 = datetime(2016, 1, 1)
   date2 = datetime(2016, 1, 2)
   for file, time in instrument_A.find_files(date1, date2, sort=True):
      print("File: {}\n\tStart: {}, End: {}".format(file, time[0], time[1]))

Output:

..

   File: Data/InstrumentA/2016/01/01/data_00_00_00.nc
       Start: 2016-01-01 00:00:00, End: 2016-01-01 00:00:00
   File: Data/InstrumentA/2016/01/01/data_06_00_00.nc
       Start: 2016-01-01 06:00:00, End: 2016-01-01 06:00:00
   File: Data/InstrumentA/2016/01/01/data_12_00_00.nc
       Start: 2016-01-01 12:00:00, End: 2016-01-01 12:00:00
   File: Data/InstrumentA/2016/01/01/data_18_00_00.nc
       Start: 2016-01-01 18:00:00, End: 2016-01-01 18:00:00

The :meth:`typhon.spareice.datasets.Dataset.find_files` method find all
files between two dates and returns their names and time coverages (start
and end times). If we want to sort them by their starting times, we the
*sort* parameter to true.

Read and Create Files
---------------------

The Dataset class has more interesting functionality that we are going to
investigate in more detail later. But before doing this, we have to understand
 how
we can open and read files from one dataset. Since there are a lot of
different types of datasets out there and each one of them might have its own
file format, the Dataset object needs help from you in order to be able to
handle those files. You must tell the Dataset how to read and write its
files by giving a *file handler* to it. A file handler is an object that
can read a file in a certain format or write data to it. For example, we
want to read the files from our Instrument A and print out their content, we
need a file handler that can handle those files. The files are stored in the
NetCDF4 format. Lucky for us, there is a file handler class that can handle
such files (:class:`typhon.spareice.handlers.commom.NetCDF4`, for a complete
list of official handler classes in typhon have a look at TODO). The only
thing that we need to do now, is give this file handler object to the
dataset object during initialization:

.. code-block:: python

   # Import the Dataset class from the typhon module.
   from typhon.spareice.datasets import Dataset
   from typhon.spareice.handlers.common import NetCDF4

   # Define a dataset object with the files.
   instrument_A = Dataset(
      name="InstrumentA",
      files="Data/InstrumentA/{year}/{month}/{day}/"
            "data_{hour}-{minute}-{second}.nc",
      # With the next line, the dataset object knows how to handle its files:
      handler=NetCDF4(),
   )




FileHandler.write(...)
++++++++++++++++++++++

FileHandler.get_info(...)
+++++++++++++++++++++++++

Get the time coverage by filename or content
--------------------------------------------


.. _sec-placeholders:

Placeholders
++++++++++++



Iterating over files in a period
--------------------------------

When you add a new function or class, you also have to add its name the
corresponing rst file in the doc/ folder.

Via .find_files(...)
++++++++++++++++++++

All code documentation in `Typhon` should follow the Google Style Python
Docstrings format. Below you can find various example on how the docstrings
should look like. The example is taken from
http://sphinxcontrib-napoleon.readthedocs.org/en/latest/example_google.html

Download: :download:`example_google.py <example_google.py>`

.. literalinclude:: example_google.py
   :language: python

Via .read_period(...)
+++++++++++++++++++++



Via .map(...) or .map_content(...)
++++++++++++++++++++++++++++++++++

All documentation for properties should be attached to the getter function
(@property). No information should be put in the setter function of the
property. Because all access occurs through the property name and never by
calling the setter function explicitly, documentation put there will never be
visible. Neither in the ipython interactive help nor in Sphinx.

Via magic indexing
++++++++++++++++++


Find overlapping files between two datasets
-------------------------------------------
