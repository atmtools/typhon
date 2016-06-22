Developer documentation
=======================

Coding Style
------------

Overall code formatting should adhere to the `Google Python Style Rules`_. Most
notably, use 4 spaces for indentation (no tabs!) and try to keep maximum line
length to 80 characters.

.. _`Google Python Style Rules`: https://google.github.io/styleguide/pyguide.html

Documentation Style
-------------------

All code documentation in `Typhon` should follow the Google Style Python
Docstrings format. Below you can find various example on how the docstrings
should look like. The example is taken from
http://sphinxcontrib-napoleon.readthedocs.org/en/latest/example_google.html

Download: :download:`example_google.py <example_google.py>`

.. literalinclude:: example_google.py
   :language: python

Common module names
-------------------

This is a list of short names that should be used consistently for importing
external modules::

  import numpy as np
  import scipy as sp
  import matplotlib as mpl
  import matplotlib.pyplot as plt

