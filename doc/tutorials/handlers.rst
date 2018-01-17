How to create your own file handler
###################################

**TODO: Continue tutorial**

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