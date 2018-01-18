How to use typhon without running X server?
===========================================

Typhon uses matplotlib_ both explicitly for plotting and implicitly in various
packages (e.g. pandas_, xarray_).

Matplotlib provides different backends to create figures. The default
backend requires a running `X server`_.

If you want to use typhon without X server (e.g. on a remote system) there
are several ways to change the backend manually.

Change backend after import
---------------------------

You can change the backend for a single python script directly after importing
matplotlib. It is important to change the backend **before any other imports!**

.. code-block:: python

    import matplotlib
    matplotlib.use('agg')

    # Other imports...

Command line flag
-----------------

The backend for interactive IPython_ sessions can be passed through the command
line flag ``--pylab``:

.. code-block:: bash

    $ ipython --pylab=agg

Matplotlib configuration
------------------------

Alternately, you can set the backend parameter in your `.matplotlibrc`_
file to automatically have matplotlib use the given backend::

    backend: agg


Environment variable
--------------------

When invoking python on the command line you can also control the backend
through an environment variable in your shell.

.. code-block:: bash

    $ export MPLBACKEND=agg


.. _IPython: https://ipython.org/
.. _matplotlib: https://matplotlib.org/
.. _`.matplotlibrc`: https://matplotlib.org/users/customizing.html#the-matplotlibrc-file
.. _pandas: https://pandas.pydata.org/
.. _xarray: http://xarray.pydata.org/en/stable/
.. _`X server`: https://en.wikipedia.org/wiki/X_Window_System
