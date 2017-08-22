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

from typhon.arts.workspace.api       import arts_api, variable_value_factory
from typhon.arts.workspace.methods   import WorkspaceMethod, workspace_methods
from typhon.arts.workspace.variables import WorkspaceVariable, group_names, group_ids
from typhon.arts.workspace.agendas   import Agenda
from typhon.arts.workspace import variables as V

imports = dict()

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
        vars(dict): Dictionary holding anonymous variables that are not shared with
                    other workspaces and have been create using the corresponding Create
                    ARTS WSM.


    """
    def __init__(self):
        """
        The init function just creates an instance of the ArtsWorkspace class of the
        C API and sets the ptr attributed to the returned handle.

        It also adds all workspace methods as attributes to the object.
        """
        self.ptr     = arts_api.create_workspace()
        self.workspace_size = arts_api.get_number_of_variables()
        self.vars = dict()
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
        an AnonymousValue object as a handle to this value in the ARTS workspace. Currently
        supported types are int, str, and numpy.ndarrays, which will automatically converted
        to the corresponding values ARTS groups.

        The user should not have to call this method explicitly, but instead it is used by
        the WorkspaceMethod call function to transfer python variable arguments to the
        ARTS workspace.

        Args:
            var: Python variable of type int, str or np.ndarray which should be copied to
            the workspace.
        """
        if type(var) == WorkspaceVariable:
            return var
        group_id = WorkspaceVariable.get_group_id(var)
        s  = variable_value_factory(var)
        ws_id = arts_api.add_variable(self.ptr, group_id, s)
        arts_api.set_variable_value(self.ptr, ws_id, group_id, s)
        return WorkspaceVariable(ws_id,
                                 str(id(var)),
                                 group_names[group_id],
                                 "User defined variable.",
                                 self)

    def __getattr__(self, name):
        """ Lookup the given variable in the anonymous variables and the ARTS workspace.

        Args:
            name(str): Name of the attribute (variable)

        Raises:
            ValueError: If the variable is not found.
        """

        group_id = None
        if name in self.vars:
            var = self.vars[name]
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

    def execute_controlfile(self, name):
        """ Execute a given controlfile on the workspace.

        This method looks recursively for a controlfile with the given name in the current
        directory and the arts include path. If such a file has been found it will be parsed
        and executed on the workspace.

        This methods takes also care of keeping the indices of anonymous variables in the
        workspace consistent, which may change if the controlfile introduces new workspace
        variables.

        Args:

            name(str): Name of the controlfile

        Raises:

            Exception: If parsing of the controlfile fails.

        """
        if not name in imports:
            try:
                imports[name] = Agenda.parse(name)
            except:
                Exception("Error parsing controlfile " + name )


        imports[name].execute(self)

        # Update workspace IDs of anonymous variables, which is necessary since executing
        # an agenda may require increasing the space reserved for ARTS WSVs.
        old_size = self.workspace_size
        self.workspace_size = arts_api.get_number_of_variables()
        new_variables = self.workspace_size - old_size
        for name in self.vars:
            self.vars[name].ws_id += new_variables
