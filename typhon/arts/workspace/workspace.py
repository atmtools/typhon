"""
The workspace submodule.

Contains the Workspace which implements the main functionality of the ARTS interface.
Users should only have to use this class to interact with ARTS.

Attributes:
     imports(dict): Dictionary of parsed controlfiles. This is kept to ensure to avoid
                    crashing of the ARTS runtime, when a file is parsed for the second time.

"""
import ctypes as c
import numpy  as np

import ast
from   ast      import iter_child_nodes, parse, NodeVisitor, Call, Attribute, Name, \
                       Expression, FunctionDef
from   inspect  import getsource, getsourcelines
from contextlib import contextmanager
from copy       import copy
from functools  import wraps
import os

from typhon.arts.workspace.api       import arts_api, VariableValueStruct, \
                                            data_path_push, data_path_pop, \
                                            include_path_push, include_path_pop
from typhon.arts.workspace.methods   import WorkspaceMethod, workspace_methods
from typhon.arts.workspace.variables import WorkspaceVariable, group_names, group_ids, \
                                            workspace_variables
from typhon.arts.workspace.agendas   import Agenda
from typhon.arts.workspace import variables as V

imports = dict()


################################################################################
# ARTS Agenda Macro
################################################################################
def arts_agenda(func):
    """
    Parse python method as ARTS agenda

    This decorator can be used to define ARTS agendas using python function syntax.
    The function should have one arguments which is assumed to be a Workspace instance.
    All expressions inside the function must be calls to ARTS WSMs. The result is an
    Agenda object that can be used to copied into a named ARTS agenda

    Example:

    >>> @arts_agenda
    >>> def inversion_iterate_agenda(ws):
    >>>     ws.x2artsStandard()
    >>>     ws.atmfields_checkedCalc()
    >>>     ws.atmgeom_checkedCalc()
    >>>     ws.yCalc()
    >>>     ws.VectorAddVector(ws.yf, ws.y, ws.y_baseline)
    >>>     ws.jacobianAdjustAfterIteration()
    >>>
    >>> ws.Copy(ws.inversion_iterate_agenda, inversion_iterate_agenda)
    """
    source = getsource(func)
    ast = parse(source)

    func_ast = ast.body[0]
    if not type(func_ast) == FunctionDef:
        raise Exception("ARTS agenda definition can only decorate function definiitons.")

    args = func_ast.args.args

    try:
        arg_name = func_ast.args.args[0].arg
    except:
        raise Exception("Agenda definition needs workspace arguments.")

    ws = Workspace()

    context = copy(func.__globals__)
    context.update({arg_name : ws})

    # Create agenda
    a_ptr = arts_api.create_agenda(func.__name__.encode())
    agenda = Agenda(a_ptr)

    for e in func_ast.body:
        if not type(e.value) == Call:
            raise Exception("Agendas may only contain call expressions.")

        # Extract workspace object.
        try:
            call = e.value
            att  = call.func.value
            if not att.id == arg_name:
                raise(Exception("Agenda definition may only contain call to WSMs of the "
                                + "workspace argument " + arg_name + "."))
        except:
            raise(Exception("Agenda definition may only contain call to WSMs of the "
                                + "workspace argument " + arg_name + "."))

        # Extract method name.
        try:
            name = call.func.attr
            m    = workspace_methods[name]
            if not type(m) == WorkspaceMethod:
                raise Exception(name + " is not a known WSM.")
        except:
            raise Exception(name + " is not a known WSM.")

        # Extract positional arguments
        args = [ws, m]
        for a in call.args:
            args.append(eval(compile(Expression(a), "<unknown>", 'eval'), context))

        # Extract keyword arguments
        kwargs = dict()
        for k in call.keywords:
            kwargs[k.arg] = eval(compile(Expression(k.value), "<unknown>", 'eval'), context)

        # Add function to agenda
        agenda.add_method(*args, **kwargs)
    return agenda


################################################################################
# Workspace Method Wrapper Class
################################################################################
class WSMCall:
    """
    Wrapper class for workspace methods. This is necessary to be able to print
    the method doc as __repr__, which doesn't work for python function objects.

    Attributes:

        ws: The workspace object to which the method belongs.
        m:  The WorkspaceMethod object

    """
    def __init__(self, ws, m):
        self.ws = ws
        self.m  = m
        self.__doc__  = m.__doc__

    def __call__(self, *args, **kwargs):
        self.m.call(self.ws, *args, **kwargs)

    def __repr__(self):
        return repr(self.m)

################################################################################
# The Workspace Class
################################################################################
class Workspace:
    """
    The Workspace class represents an ongoing ARTS simulation. Each Workspace object
    holds its own ARTS workspace and can be used to execute ARTS workspace methods or
    access workspace variables.

    All workspace methods taken from workspace_methods in the methods module are added
    as attributed on creation and are thus available as class methods.

    Attributes:

        ptr(ctypes.c_void_p): object pointing to the ArtsWorkspace instance of the
        ARTS C API
        _vars(dict): Dictionary holding local variables that have been created
                    interactively using the one of Create ARTS WSMs.


    """
    def __init__(self):
        """
        The init function just creates an instance of the ArtsWorkspace class of the
        C API and sets the ptr attributed to the returned handle.

        It also adds all workspace methods as attributes to the object.
        """

        self.__dict__["_vars"] = dict()
        self.ptr     = arts_api.create_workspace()
        self.workspace_size = arts_api.get_number_of_variables()
        for name in workspace_methods:
            m = workspace_methods[name]
            setattr(self, m.name, WSMCall(self, m))

    def __del__(self):
        """
        Cleans up the C API.
        """
        if (arts_api):
            arts_api.destroy_workspace(self.ptr)

    def add_variable(self, var):
        """
        This will try to copy a given python variable to the ARTS workspace and
        return a WorkspaceVariable object representing this newly created
        variable.

        Types are natively supported by the C API are int, str, [str], [int], and
        numpy.ndarrays. These will be copied directly into the newly created WSV.

        In addition to that all typhon ARTS types the can be stored to XML can
        be set to a WSV, but in this case the communication will happen through
        the file systs (cf. WorkspaceVariable.from_typhon).

        The user should not have to call this method explicitly, but instead it
        is used by the WorkspaceMethod call function to transfer python
        variable arguments to the ARTS workspace.

        Args:
            var: Python variable of type int, str, [str], [int] or np.ndarray
            which should be copied to the workspace.
        """
        if type(var) == WorkspaceVariable:
            return var

        # Create WSV in ARTS Workspace
        group_id = WorkspaceVariable.get_group_id(var)
        ws_id    = arts_api.add_variable(self.ptr, group_id, None)
        wsv      = WorkspaceVariable(ws_id,
                                     str(id(var)),
                                     group_names[group_id],
                                     "User defined variable.",
                                     self)
        # Set WSV value using the ARTS C API
        s  = VariableValueStruct(var)
        if s.ptr:
            e = arts_api.set_variable_value(self.ptr, ws_id, group_id, s)
            if e:
                arts_api.erase_variable(self.ptr, ws_id, group_id)
                raise Exception("Setting of workspace variable through C API "
                                " failed with  the " + "following error:\n"
                                + e.decode("utf8"))
        # If the type is not supported by the C API try to write the type to XML
        # and read into ARTS workspace.
        else:
            try:
                wsv.from_typhon(var)
            except:
                raise Exception("Could not add variable since + "
                                + str(type(var)) + " is neither supported by "
                                + "the C API nor typhon XML IO.")
        return wsv

    def __dir__(self):
        return {**workspace_variables, **self.__dict__}

    def __getattr__(self, name):
        """ Lookup the given variable in the local variables and the ARTS workspace.

        Args:
            name(str): Name of the attribute (variable)

        Raises:
            ValueError: If the variable is not found.
        """

        group_id = None
        if name in self._vars:
            var = self._vars[name]
            var.update()
            return var
        else:
            i = arts_api.lookup_workspace_variable(name.encode())
            if i < 0:
                raise ValueError("No workspace variable " + str(name) + " found.")
            vs = arts_api.get_variable(i)
            group_id    = vs.group
            description = vs.description.decode("utf8")

        # Get its symbolic representation
        wsv = WorkspaceVariable(i, name, group_names[group_id], description, self)
        return wsv

    def __setattr__(self, name, value):
        """ Set workspace variable.

        This will lookup the workspace variable name and try to set it to value.

        Args:
            name(str):  Name of the attribute (variable)
            value(obj): The value to set the workspace variable to.

        Raises:
            ValueError: If the variable is not found or if value cannot uniquely converted to
            a value of a workspace variable.
        """
        try:
            v = self.__getattr__(name)
        except:
            self.__dict__[name] = value
            return None

        t = self.add_variable(value)

        if not t.group_id == v.group_id:
            raise Exception("Incompatible groups: Workspace variable " + name +
                            " and value " + str(value))

        self.Copy(v, t)

        # Remove t only if it wasn't an existing WSV already before.
        if not type(value) == WorkspaceVariable:
            t.erase()

    def execute_controlfile(self, name):
        """ Execute a given controlfile on the workspace.

        This method looks recursively for a controlfile with the given name in the current
        directory and the arts include path. If such a file has been found it will be parsed
        and executed on the workspace.

        Args:

            name(str): Name of the controlfile

        Raises:

            Exception: If parsing of the controlfile fails.

        """
        if not name in imports:
            imports[name] = Agenda.parse(name)

        include_path_push(os.getcwd())
        data_path_push(os.getcwd())

        imports[name].execute(self)

        include_path_pop()
        data_path_pop()
