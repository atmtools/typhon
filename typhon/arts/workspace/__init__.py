"""

The workspace subpackage provides an interactive interface to ARTS and
can be used to directly call workspace methods and agendas as well as
access and manipulate workspace variables.

Setup
=====

The python interface needs access to the arts_api.so shared library, which is located in
the src subfolder of the ARTS build tree. The interface expects the location of the ARTS
build tree to be provided in the ``ARTS_BUILD_PATH`` environment variable.

The Workspace Class
===================

The main functionality of the interface is implemented by the Workspace class. A
Workspace object represents an ongoing ARTS simulation and is used to execute controlfiles
and workspace methods and access workspace variables

    >>> from typhon.arts.workspace import Workspace
    >>> ws = Workspace()

Executing Controlfiles
======================

Controlfiles can be execute on a workspace using the execute_controlfile class method.

    >>> ws.execute_controlfile("general.arts")

This will search recursively through the working directory and the path provided in
the environment variable ``ARTS_INCLUDE_PATH`` and parse and execute the found controlfile
on the workspace object.

Parsing and execution of controlfiles is handled completely by ARTS, so in order to resolve
INCLUDE statements and loading of data the include path and the data path must be set correctly.
By default, the ARTS interface will set them to be the controlfiles subfolder of the path
contained in ``ARTS_INCLUDE_PATH`` and the controlfiles/testdata folder, respectively. Both can
be set manually using the include_path_add and data_path_add methods.

Calling Workspace Methods
=========================

ARTS workspace methods are available as member functions of each Workspace object:

    >>> ws.AtmosphereSet1D()
    >>> ws.IndexSet(ws.stokes_dim, 1)

Arguments can be passed to a workspace function in three ways:

    1. As workspace variables using the attributes of the workspace object (ws.stokes_dim)
    2. Using one of the symbolic variables in typhon.arts.workspace.variables
    3. Passing python objects directly

Arguments to a WSM can be passed using either positional or named arguments or both. If
positional arguments are provided at least all generic output and generic input arguments
must be given in order.

    >>> ws.VectorNLogSpace(ws.p_grid, 361, 500e2, 0.1 )

Keyword arguments to define generic input and output arguments. Available keywords are the
names of the generic outputs and inputs defined in methods.cc.

    >>> ws.abs_speciesSet(species=[ "O3" ])

Calls to supergeneric functions are resolved by the interface.

Workspace Variables
===================

Symbolic representation of all workspace variables are available in the typhon.arts.workspace.variables
module as module attributes. The purpose of these is to be passed to workspace functions as placeholders
for variables in the workspace.

Variable objects can be associated to a workspace, which is the case for variables accessed as attributes
of a workspace, i.e. using for example:

    >>> ws.y

If that is the case their value can be accessed using the value() member function. In order
to print out a textual representation of the value, the print() member function can be used,
which will call the corresponding Print() ARTS WSV.

Workspace variables of the groups Vector, Matrix and Tensor with an associated workspace
implement the numpy array interface and can therefore be used just as any other numpy array.
In some cases, however, it may be necessary to explicitly create a view on the array using
numpy.asarray.

"""

from os import environ
from warnings import warn

if environ.get('ARTS_BUILD_PATH') is None:
    warn("ARTS_BUILD_PATH environment variable required to locate ARTS API.")
else:
    from typhon.arts.workspace.workspace import Workspace, arts_agenda
    from typhon.arts.workspace.variables import WorkspaceVariable
    from typhon.arts.workspace.methods   import WorkspaceMethod
    from typhon.arts.workspace.api       import arts_include_path, include_path_add, data_path_add

