""" The workspace submodule.

Contains the Workspace which implements the main functionality of the ARTS interface.
Users should only have to use this class to interact with ARTS.

Attributes:
     imports(dict): Dictionary of parsed controlfiles. This is kept to ensure to avoid
                    crashing of the ARTS runtime, when a file is parsed for the second time.

"""
import ctypes as c
import numpy  as np
import os

import ast
from   ast     import iter_child_nodes, parse, NodeVisitor, Call, Attribute, Name, \
                      Expression, FunctionDef
from   inspect import getsource, getsourcelines

from typhon.arts.workspace.api       import arts_api, VariableValueStruct
from typhon.arts.workspace.methods   import WorkspaceMethod, workspace_methods
from typhon.arts.workspace.variables import WorkspaceVariable, group_names, group_ids, \
                                            workspace_variables
from typhon.arts.workspace.agendas   import Agenda
from typhon.arts.workspace import variables as V

imports = dict()


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

    context = func.__globals__
    context[arg_name] == ws

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
            def make_fun(method):
                return lambda *args, **kwargs: method.call(self, *args, **kwargs)
            setattr(self, m.name, make_fun(m))
            getattr(self, m.name).__doc__ = m.description

    def __del__(self):
        """
        Cleans up the C API.
        """
        if (arts_api):
            arts_api.destroy_workspace(self.ptr)

    def add_variable(self, var):
        """
        This will try to copy a given python variable to the ARTS workspace and return
        a WorkspaceVariable object representing this newly created variable. Currently
        supported types are int, str, [str], [int], and numpy.ndarrays, which will
        automatically converted to the corresponding values ARTS groups.

        The user should not have to call this method explicitly, but instead it is used by
        the WorkspaceMethod call function to transfer python variable arguments to the
        ARTS workspace.

        Args:
            var: Python variable of type int, str, [str], [int] or np.ndarray which should
                 be copied to the workspace.
        """
        if type(var) == WorkspaceVariable:
            return var
        group_id = WorkspaceVariable.get_group_id(var)
        s  = VariableValueStruct(var)
        ws_id = arts_api.add_variable(self.ptr, group_id, None)
        arts_api.set_variable_value(self.ptr, ws_id, group_id, s)
        return WorkspaceVariable(ws_id,
                                 str(id(var)),
                                 group_names[group_id],
                                 "User defined variable.",
                                 self)

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

        try:
            t = self.add_variable(value)
        except:
            raise Exception("Given value " + str(value) + " could not be uniquely converted "
                            "to ARTS value." )

        if not t.group_id == v.group_id:
            raise Exception("Incompatible groups: Workspace variable " + name +
                            " and value " + str(value))

        fname = v.group + "Set"
        workspace_methods[fname].call(self, v, t)
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
            try:
                imports[name] = Agenda.parse(name)
            except:
                raise Exception("Error parsing controlfile " + name )

        imports[name].execute(self)
