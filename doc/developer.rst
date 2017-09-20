Developer documentation
=======================

Coding Style
------------

Overall code formatting should adhere to the `Google Python Style Rules`_. Most
notably, use 4 spaces for indentation (no tabs!) and try to keep maximum line
length to 80 characters.

.. _`Google Python Style Rules`: https://google.github.io/styleguide/pyguide.html

Comments
--------

All code should be properly commented. To quote a `blog post`_ on dev.to:

    Comments aren't additional to a good codebase. They are the codebase.

While clean code shows **what** is done, comments should explain **why** a
feature is implemented in a certain way. This approach is often called
*Commenting Showing Intent* (`CSI`_). It helps other developers and your future
self to comprehend why a certain implementation has been chosen. In addition,
possible implications with other parts of the project should be addressed.

.. _`CSI`: https://standards.mousepawmedia.com/csi.html
.. _`blog post`: https://dev.to/andreasklinger/comments-explain-why-not-what-and-2-more-rules-on-writing-good-comments

Documentation Style
-------------------

General
+++++++

All code documentation in `Typhon` should follow the Google Style Python
Docstrings format. Below you can find various example on how the docstrings
should look like. The example is taken from
http://sphinxcontrib-napoleon.readthedocs.org/en/latest/example_google.html

Download: :download:`example_google.py <example_google.py>`

.. literalinclude:: example_google.py
   :language: python

Properties
++++++++++

All documentation for properties should be attached to the getter function
(@property). No information should be put in the setter function of the
property. Because all access occurs through the property name and never by
calling the setter function explicitly, documentation put there will never be
visible. Neither in the ipython interactive help nor in Sphinx.

Adding functions / classes
--------------------------

When you add a new function or class, you also have to add its name the
corresponing rst file in the doc/ folder.

Common module names
-------------------

This is a list of short names that should be used consistently for importing
external modules::

  import numpy as np
  import scipy as sp
  import matplotlib as mpl
  import matplotlib.pyplot as plt

