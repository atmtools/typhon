Using typhon.spareice.Dataset for data processing
#################################################

.. contents:: :local:

.. highlight:: python
   :linenothreshold: 5


What is the idea?
=================

Imagine you have a big dataset consisting of many files containing observations
(e.g. images or satellite data). The files cover certain time periods and
are bundled into subdirectories. See
:numref:`Fig.{number}<fig-example-datasets>` for an example.

.. _fig-example-datasets:

.. figure:: _figures/example_datasets.png
   :scale: 50 %
   :alt: screen shot of dataset directory structure

   Example of datasets

   All files of *Satellite B* are located in subdirectories which
   contain temporal information in their names (year, month, day, etc.).

Typical tasks to analyze this dataset would include iterating over those
files, finding those that cover a certain time period, reading them, applying
functions on their content and eventually adding files with new data to this
dataset. So, how to find all files in a time period? You could start by writing
nested *for* loops and using python's *glob* function. Normally, such solutions
requires time to implement, are error-prone and are not portable to other
datasets with different structures. Hence, save your time/energy/nerves and
simply use the :class:`~typhon.spareice.datasets.Dataset` class.

.. Hint::
   If you want to run the code from this tutorial on your machine as well,
   download
   :download:`spareice_tutorials.zip<_downloads/spareice_tutorials.zip>` and
   unzip it. You can find the code examples for this tutorial in the jupyter
   notebook file *dataset.ipynb*. You will need the jupyter_ engine for this.

.. _jupyter: http://jupyter.org/install.html

Find Files
==========

We stick to our example from above and want to find all files from our
*Satellite B* dataset between two dates. To do this, we have to create a
Dataset object with the path to our files:

.. code-block:: python

   # Import the Dataset class from the typhon module.
   from typhon.spareice import Dataset

   # Define a dataset object pointing to the files
   # of the Satellite B
   b_dataset = Dataset(
       path="data/SatelliteB/{year}/{month}/{day}/"
            "{hour}{minute}{second}-{end_hour}{end_minute}{end_second}.nc.gz"
   )

Nothing interesting happens so far. We imported the Dataset class from the 
typhon module, created a Dataset object and told it where to find its files.
These words surrounded by braces (e.g. "{year}") are called placeholders. They
work like regular expressions and generalize the path, so we need not give 
explicit paths that point to the files directly. The Dataset object can fill
those placeholders by itself when searching for files. Let's see it in action:

.. code-block:: python

    # Find all files between 2018-01-01 and 2018-01-02:
    for file in b_dataset.find("2018-01-01", "2018-01-02"):
        print(file)

.. code-block:: none
   :caption: Output:

   .../data/SatelliteB/2018/01/01/000000-060000.nc.gz
      Start: 2018-01-01 00:00:00
      End: 2018-01-01 06:00:00
   .../data/SatelliteB/2018/01/01/060000-120000.nc.gz
      Start: 2018-01-01 06:00:00
      End: 2018-01-01 12:00:00
   .../data/SatelliteB/2018/01/01/120000-180000.nc.gz
      Start: 2018-01-01 12:00:00
      End: 2018-01-01 18:00:00
   .../data/SatelliteB/2018/01/01/180000-000000.nc.gz
      Start: 2018-01-01 18:00:00
      End: 2018-01-02 00:00:00

The :meth:`~typhon.spareice.datasets.Dataset.find` method finds all
files between two dates and returns their names with some further information
in :class:`~typhon.spareice.handlers.common.FileInfo` objects. The FileInfo
object has three attributes: *path*, *times* and *attr*. Let's have a look at
them:

.. code-block:: python

   print("Path:", file.path)
   print("Times:", file.times)
   print("Attributes", file.attr)

.. code-block:: none
   :caption: Output:

   Path: .../data/SatelliteB/2018/01/01/000000-060000.nc.gz
   Times: [datetime.datetime(2018, 1, 1, 0, 0), datetime.datetime(2018, 1, 1, 6, 0)]
   Attributes: {}

Surprisingly, *path* returns the path to the file and *times* is a list with
two datetime objects: the start and end time of the file. They are retrieved by
the placeholders that were used in the *path* argument of the Dataset object.
But what is about *attr* and why is it an empty dictionary? Additionally to the
temporal placeholders (such as {year}, etc.), which are converted into start
and end datetime objects, you can define own placeholders. For example, let's
make a placeholder out of the satellite name:

.. code-block:: python

   # The same dataset as before but with one additional placeholder in the
   # path:
   dataset = Dataset(
      path="data/{satname}/{year}/{month}/{day}/"
           "{hour}{minute}{second}-{end_hour}{end_minute}{end_second}.nc.gz"
   )

   for file in dataset.find("01/01/2018", "2018-01-02"):
      print("Path:", file.path)
      print("Attributes", file.attr)

.. code-block:: none
   :caption: Output:

   Path: .../data/SatelliteA/2018/01/01/000000-050000.nc.gz
      Attributes {'satname': 'SatelliteA'}
   Path: .../data/SatelliteB/2018/01/01/000000-060000.nc.gz
      Attributes {'satname': 'SatelliteB'}

As we can see, we are able to find the data from *Satellite A* as well because
it has the same subdirectory structure as *Satellite B*. The placeholder
*satname* - per default interpreted as wildcard - was filled by Dataset
automatically and returned in *attr*. This could be useful if we want to
process our files and we need to know from which satellite they came from. We
can apply a filter on this placeholder when using
:meth:`~typhon.spareice.datasets.Dataset.find`:

.. code-block:: python

   filters = {"satname": "SatelliteA"}
   for file in dataset.find("2018-01-01", "2018-01-02", filters=filters):
       print("Path:", file.path)
       print("  Attributes", file.attr)

This finds only the files which placeholder *satname* is *SatelliteA*. We can
also set it to a regular expression. If we want to apply our filter as a black
list, i.e. we want to skip all files which placeholders contain certain values,
we can add a *!* before the placeholder name.

.. code-block:: python

   # This finds all files which satname is not SatelliteA
   filters = {"!satname": "SatelliteA"}

We can also set a placeholder permanently to our favourite regular expression
(e.g. if you want to call :meth:`~typhon.spareice.datasets.Dataset.find`
multiple times). Use
:meth:`~typhon.spareice.datasets.Dataset.set_placeholders` for this:

.. code-block:: python

   dataset.set_placeholders(satname="\w+?B")

Which results that we only find satellites which name ends with *B*. If you
want to find out more about placeholders, have a look at this
:ref:`section<typhon-dataset-placeholders>`.


Read and Create Files
=====================

Handling common file formats
++++++++++++++++++++++++++++

Well, it is nice to find all files from one dataset. But we also want to open
them and read their content. For doing this, we could use our found `FileInfo`
objects as file argument for python's `open` builtin function:

.. code-block:: python

   for file in b_dataset.find("2018-01-01", "2018-01-02"):
      with open(file, "rb") as f:
         # This returns a lot of byte strings:
         print(f.readline())

Okay, this may be not very practical for gzipped netCDF files since it just
returns a lot of byte strings. Of course, we could use the `python-netcdf`
module for reading the files but then we would still need to unzip them by
ourselves before. Well, we could do that. But our Dataset object provides
a much easier way:

.. code-block:: python

   data = b_dataset["2018-01-01"]
   print(data)

.. code-block:: none
   :caption: Output:

   Name: 120729074544 <class 'typhon.spareice.array.ArrayGroup'>
   Attributes:
      --
   Groups:
      --
   Variables:
      lat (40,) :
      [-0.00159265 -0.16190251 -0.31802342 -0.46591602 -0.60175384 -0.72202232
       -0.82360972 -0.90388763 -0.96077901 -0.99281188 -0.99915745 -0.97965155
       -0.93479885 -0.86575984 -0.77432078 -0.66284751 -0.5342242  -0.39177875
      ...

This found a file that is the closest to 2018-01-01 and decompressed it.
Afterwards it loaded its decompressed content into an
:class:`~typhon.spareice.array.ArrayGroup` object (kind of dictionary that
holds numpy arrays). And all this by using only one single expression! We can
also read all files from a time period:

.. code-block:: python

   # Find files from 2018-01-01 to 2018-01-01 and load them into
   # numpy arrays
   data = dataset["2018-01-01":"2018-01-02"]

   # data is now a list of ArrayGroup objects.

What if we want to create a new file for our Dataset? How does this work? It
is as simple as reading them. Create your data object first and then pass it to
the Dataset:

.. code-block:: python

   import numpy as np
   from typhon.spareice import Array, ArrayGroup

   # Create an ArrayGroup which holds data in form of numpy arrays. This should
   # work with xarray.Dataset as well.
   data = ArrayGroup()
   data['lat'] = Array(
       90*np.sin(np.linspace(0, 6.28, 7)),
       dims=('time',)
   )
   data['lon'] = Array(
       np.linspace(-180, 180, 7), dims=('time',)
   )
   data['data'] = Array(
       data['lat'] * 2 + np.random.randn(7), dims=('time',)
   )
   data["time"] = np.arange(
       "2018-01-03 06:00:00", "2018-01-03 13:00:00",
       dtype="datetime64[h]"
   )

   # Save this ArrayGroup object to a file that belongs to our Dataset:
   dataset["2018-01-03 06:00:00":"2018-01-03 12:00:00"] = data

If we look now in our dataset directory, we find a new file called
*data/SatelliteB/2018/01/03/060000-120000.nc.gz*. We can unzip it and see its
content with a netCDF viewer (e.g. panoply). So our Dataset object took
our ArrayGroup, put it into a netCDF file and gzipped it automatically. The
Dataset object tries to detect from the path suffix the format of the files.
This works for netCDF files (*\*.nc*) and for CSV files (*\*.txt*, *\*.asc* or
*\*.csv*).

Handling other file formats
+++++++++++++++++++++++++++

What happens with files in CSV format but with a different filename suffix? Or
with other file formats, e.g. such as from CloudSat instruments? Can the
Dataset read and write them as well? Yes, it can. But it is going to need some
help of us before doing so. To understand this better, we have to be honest:
the Dataset object cannot do very much; it simply finds files and compress /
decompress them if necessary. However, to read or create files, it trusts a
*file handler* and let it do the format-specific work. A file handler is an
object, which knows everything about a certain file format and hence can read
it or use it to write a new file. The Dataset object automatically tries to
find an adequate file handler according to the filename suffix. Hence, it knew
that our files from *Satellite B* (with the suffix *.nc.gz*) have to be
decompressed and then opened with the
:class:`~typhon.spareice.handlers.common.NetCDF4` file handler.

If we want to use another file handler, we can set the file handler by
ourselves. Let's demonstrate this by using another dataset, e.g. data from
*Satellite C*. Its structure looks like this:

.. _fig-example-dataset_c:

.. figure:: _figures/example_dataset_c.png
   :scale: 50 %
   :alt: screen shot of dataset directory structure

   Files of Satellite C

The files are stored in a different directory structure and are
in CSV format. Instead of having subdirectories with month and day, we now have
subdirectories with the so-called day-of-year (all days since the start of the
year). Do not worry, the Dataset object can handle this structure without any
problems:

.. code-block:: python

   c_dataset = Dataset(
      path="data/SatelliteC/{year}/{doy}/{hour}{minute}{second}.dat.gz",
   )

   for file in c_dataset.find("2018-01-01", "2018-01-02"):
      print(file)

.. code-block:: none
   :caption: Output

   .../data/SatelliteC/2018/001/000000.dat.gz
      Start: 2018-01-01 00:00:00
      End: 2018-01-01 00:00:00
   ...

But if we try to open one of the files, the following happens:

.. code-block:: python

   data = c_dataset["2018-01-01"]

.. code-block:: none
   :caption: Output

   ---------------------------------------------------------------------------
   NoHandlerError                            Traceback (most recent call last)
   ...

   NoHandlerError: Could not read '.../data/SatelliteC/2018/001/000000.dat.gz'!
      I do not know which file handler to use. Set one by yourself.


It cannot open the file because it could not retrieve a file handler from the
filename suffix. Let's help the Dataset object by setting its file handler to
:class:`~typhon.spareice.handlers.common.CSV` explicitly. Now it should be able
to open these CSV files.

.. code-block:: python

   from typhon.spareice.handlers import CSV

   # Create a CSV file handler that interprets the column 'time' as
   # timestamp object.
   csv_handler = CSV(
       read_csv={"parse_dates":["time", ]}
   )

   c_dataset = Dataset(
       path="data/SatelliteC/{year}/{doy}/{hour}{minute}{second}.dat.gz",
       handler=csv_handler,
   )

   c_dataset["2018-01-01"]

.. code-block:: none
   :caption: Output

   Name: 4523163040 <class 'typhon.spareice.array.ArrayGroup'>
   Attributes:
    --
   Groups:
    --
   Variables:
    Unnamed: 0 (40,) :
      [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
         24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
    data (40,) :
      ...

There are more file handlers for other file formats. For example,
:class:`~typhon.spareice.handlers.cloudsat.CloudSat` can read CloudSat HDF4
files. Have a look at :ref:`typhon-handlers` for a complete list of official
handler classes in typhon. Every file handler might have its own specifications
and options, you can read about them in their documentation.

Handling your file formats
++++++++++++++++++++++++++

If you need a special format that is not covered by the official file handlers,
you can use the generic
:class:`~typhon.spareice.handlers.common.FileHandler` object and set customized
reader and writer functions. Another way - if you like object-oriented
programming - is to subclass
:class:`~typhon.spareice.handlers.common.FileHandler` and write your own file
handler class (see :doc:`handlers` for a tutorial). Since the latter is for
more advanced programmers, here is a simple but extensive example that shows
how to use your own reader and writer functions easily. This also shows how to
create a new dataset with many files on-the-fly:

.. code-block:: python

   from datetime import datetime, timedelta

   # Get the base class to use a customized file handler
   from typhon.spareice.handlers import FileHandler


   # Here are our reader and writer functions:
   def our_reader(file_info, lineno=0):
       """Read the nth line of a text file

       Args:
           file_info: A FileInfo object.
           lineno: Number of the line that should be read.
               Default is the 0th line (header).

       Returns:
           A string with the nth line
       """

       with open(file_info, "r") as file:
           return file.readlines()[lineno]


   def our_writer(data, file_info, mode="w"):
       """Append a text to a file

       Args:
           data: A string with content.
           file_info: A FileInfo object.
           mode: The writing mode. 'w' means overwriting (default) and
               'a' means appending.

       Returns:
           A string with the first line
       """

       with open(file_info, mode) as file:
           file.write(data)

   # Let's customize a file handler with our functions
   our_handler = FileHandler(
       reader=our_reader,
       writer=our_writer,
   )

   # Let's create a new dataset and pass our file handler
   our_dataset = Dataset(
      path="data/own_dataset/{year}/{doy}/{hour}{minute}{second}.txt",
      handler=our_handler,
   )

   # Fill the dataset with files covering the next two days:
   start = datetime(2018, 1, 1)
   for hour in range(0, 48, 4):
       timestamp = start + timedelta(hours=hour)

       # The content for each file:
       text = f"Header: {timestamp}\n" \
           + "1) First line...\n" \
           + "2) Second line...\n" \
           + "3) Third line...\n"

       # Write the text to a file (uses the
       # underlying our_writer function)
       our_dataset[timestamp] = text

   # Read files at once and get their header line
   # (uses the underlying our_reader function)
   print(our_dataset["2018-01-01":"2018-01-03"])

.. code-block:: none
   :caption: Output

   ['Header: 2018-01-01 00:00:00\n', 'Header: 2018-01-01 04:00:00\n',
    'Header: 2018-01-01 08:00:00\n', 'Header: 2018-01-01 12:00:00\n',
    'Header: 2018-01-01 16:00:00\n', 'Header: 2018-01-01 20:00:00\n',
    'Header: 2018-01-02 00:00:00\n', 'Header: 2018-01-02 04:00:00\n',
    'Header: 2018-01-02 08:00:00\n', 'Header: 2018-01-02 12:00:00\n',
    'Header: 2018-01-02 16:00:00\n', 'Header: 2018-01-02 20:00:00\n']

This script creates files containing one header line with a timestamp and some
further 'data' lines. With our new file handler we can read easily the header
line from each of those files. Great!

Pass arguments to reader and writer
+++++++++++++++++++++++++++++++++++

The `our_reader` function actually provides to return the nth line of the file
if the argument `lineno` is given. If we want to read files with additional
arguments for the underlying reader function, we cannot use the simple
expression with brackets any longer. We have to use the more extended version
in form of the :meth:`~typhon.spareice.datasets.Dataset.read` method instead:

.. code-block:: python

   # Find the closest file to this timestamp:
   file = our_dataset.find_closest("2018-01-01")

   # Pass the file and the additional argument 'lineno' to the
   # underlying our_reader function:
   data = our_dataset.read(file, lineno=2)

   print(file, "\nData:", data)

.. code-block:: none
   :caption: Output

   .../data/own_dataset/2018/001/000000.txt
     Start: 2018-01-01 00:00:00
     End: 2018-01-01 00:00:00
   Data: 2) Second line...

Using additional arguments for creating a file works very similar as above, we
can use :meth:`~typhon.spareice.datasets.Dataset.write` here. Another
difference is that we have to generate a filename first by using
:meth:`~typhon.spareice.datasets.Dataset.generate_filename`.

.. code-block:: python

   # Generate a filename for our dataset and a given timestamp:
   filename = our_dataset.generate_filename("2018-01-01 04:00:00")

   data = "4) Appended fourth line...\n"

   print(f"Append {data} to {filename}")

   # Pass the data, filename and the additional argument 'mode' to
   # the underlying our_writer function:
   our_dataset.write(data, filename, mode="a")

.. code-block:: none
   :caption: Output

   Append 4) Appended fourth line...
   to .../data/own_dataset/2018/001/000000.txt

How can we read the second lines from all files? We could do this:

.. code-block:: python

   for file in our_dataset:
      data = our_dataset.read(file, lineno=2)
      ...

If you want to use parallel workers to load the files faster (will not
make much difference for our small files here though), use
:meth:`~typhon.spareice.datasets.Dataset.icollect` in combination with a
for-loop or simply :meth:`~typhon.spareice.datasets.Dataset.collect` alone:

.. code-block:: python

   # Read the second line of each file:
   for file, data in our_dataset.icollect(read_args={"lineno": 2}):
      ...

   # OR

   # Read the second line of all files at once:
   collection = our_dataset.collect(read_args={"lineno": 2})
   filenames, data_list = zip(*collection)

   print(filenames)
   print(data_list)


Get information from a file
===========================

The Dataset object needs information about each file in order to find them
properly via :meth:`~typhon.spareice.datasets.Dataset.find`. Normally, this
happens by using :ref:`placeholders<typhon-dataset-placeholders>` in the files'
path and name. Each placeholder is represented by a regular expression that is
used to parse the filename. But sometimes this is not enough. For example, if
the filename provides not the end of the file's time coverage but the file does
not represent a single discrete point. Let's have a look at our *Satellite C*
for example:

.. code-block:: python

   from typhon.spareice.handlers import CSV

   # Create a CSV file handler that interprets the column 'time' as
   # timestamp object.
   csv_handler = CSV(
       read_csv={"parse_dates":["time", ]}
   )

   c_dataset = Dataset(
       path="data/SatelliteC/{year}/{doy}/{hour}{minute}{second}.dat.gz",
       handler=csv_handler,
   )

   for file in c_dataset.find("2018-01-01", "2018-01-01 8:00:00"):
       print(file)

.. code-block:: none
   :caption: Output

   .../data/SatelliteC/2018/001/000000.dat.gz
      Start: 2018-01-01 00:00:00
      End: 2018-01-01 00:00:00
   .../data/SatelliteC/2018/001/060000.dat.gz
      Start: 2018-01-01 06:00:00
      End: 2018-01-01 06:00:00

As we can see, are files interpreted as *discrete* files: their start time is
identical with their end time. But we know that is not true, e.g.
*.../data/SatelliteC/2018/001/000000.dat.gz* covers a period from almost six
hours:

.. code-block:: python

   data = c_dataset.read("data/SatelliteC/2018/001/000000.dat.gz")
   print("Start:", data["time"].min())
   print("End:", data["time"].max())

.. code-block:: none
   :caption: Output

   Start: 2018-01-01 00:00:00
   End: 2018-01-01 05:59:59

We have two options now:

1. Use the parameter *time_coverage* of the Dataset to specify the duration per
   file. Works only if each file has the same time coverage. This is the
   easiest and fastest option.
2. Using the file handler to get more information. The file handler can more
   than only reading or creating files in a specific format. If its method
   :meth:`~typhon.spareice.handlers.common.FileHandler.get_info` is set, it can
   complement information that could not be completely retrieved from the
   filename.

Let's try at first option 1:

.. code-block:: python

   c_dataset.time_coverage = "05:59:59 hours"

   for file in c_dataset.find("2018-01-01", "2018-01-01 8:00:00"):
      print(file)

.. code-block:: none
   :caption: Output

   .../data/SatelliteC/2018/001/000000.dat.gz
      Start: 2018-01-01 00:00:00
      End: 2018-01-01 05:59:59
   .../data/SatelliteC/2018/001/060000.dat.gz
      Start: 2018-01-01 06:00:00
      End: 2018-01-01 11:59:59

It works! But what if each file has an individual duration? Then we need to
define a file handler that have a info method:

TODO


.. _typhon-dataset-placeholders:

Placeholders
============

Standard placeholders
+++++++++++++++++++++

Allowed placeholders in the *path* argument are:

+-------------+------------------------------------------+------------+
| Placeholder | Description                              | Example    |
+=============+==========================================+============+
| year        | Four digits indicating the year.         | 1999       |
+-------------+------------------------------------------+------------+
| year2       | Two digits indicating the year. [1]_     | 58 (=2058) |
+-------------+------------------------------------------+------------+
| month       | Two digits indicating the month.         | 09         |
+-------------+------------------------------------------+------------+
| day         | Two digits indicating the day.           | 08         |
+-------------+------------------------------------------+------------+
| doy         | Three digits indicating the day of       | 002        |
|             | the year.                                |            |
+-------------+------------------------------------------+------------+
| hour        | Two digits indicating the hour.          | 22         |
+-------------+------------------------------------------+------------+
| minute      | Two digits indicating the minute.        | 58         |
+-------------+------------------------------------------+------------+
| second      | Two digits indicating the second.        | 58         |
+-------------+------------------------------------------+------------+
| millisecond | Three digits indicating the millisecond. | 999        |
+-------------+------------------------------------------+------------+

.. [1] Numbers lower than 65 are interpreted as 20XX while numbers
   equal or greater are interpreted as 19XX (e.g. 65 = 1965,
   99 = 1999)

All those place holders are also allowed to have the prefix *end* (e.g.
*end_year*). They will be used to retrieve the end of the time coverage from
the filename.


User-defined placeholders
+++++++++++++++++++++++++

Further recipes
===============


Use multiple processes
++++++++++++++++++++++


Copy or convert files
+++++++++++++++++++++


Use filters with magic indexing
+++++++++++++++++++++++++++++++


Exclude or limit to time periods
++++++++++++++++++++++++++++++++

