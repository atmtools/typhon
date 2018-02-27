""" The variables submodule.

This module contains symbolic representations of all ARTS workspace variables.

The variables are loaded dynamically when the module is imported, which ensures that they
up to date with the current ARTS build.

TODO: The group names list is redudant w.rt. group_ids.keys(). Should be removed.

Attributes:
    group_names([str]): List of strings holding the groups of ARTS WSV variables.
    group_ids(dict):    Dictionary mapping group names to the group IDs which identify
                        groups in the ARTS C API.
"""

import ctypes as c
import numpy as np
import re
import scipy as sp
import tempfile

from typhon.arts.workspace.api import arts_api
from typhon.arts.workspace.agendas import Agenda
from typhon.arts.xml import load, save
from typhon.arts.xml.names import tensor_names

class WorkspaceVariable:
    """
    The WorkspaceVariable represents ARTS workspace variables in a symbolic way. This
    means that they are not associated with a single workspace and therefore do not have a
    unique value. Their value in a given workspacecan be accessed, however, using the value()
    method.

    Attributes:
    ws_id(int):          The Index variable identifying the variable in the ARTS C API.
    name(str):        The name of the workspace variable.
    group(str):       The name of the group this variable belongs to.
    description(str): The documentation of the variable as in methods.cc
    """
    def __init__(self, ws_id, name, group, description, ws = None):
        self.ws_id = ws_id
        self.name = name
        self.group = group
        self.group_id = group_ids[group]
        self.description = description
        self.ws = ws

        self.ndim = None
        if self.group == "Vector":
            self.ndim = 1
        if self.group == "Matrix":
            self.ndim = 2
        m = re.match("^Tensor(\d)$", self.group)
        if m:
            self.ndim = int(m.group(1))

        self.update()

    def __str__(self):
        return __repr__(self)

    def __repr__(self):
        s  = "ARTS Workspace Variable\n\n"
        s += "Name:  " + self.name + "\n"
        s += "Group: " + self.group + "\n\n"
        s += self.description
        return s

    def print(self):
        """ Print variable value using ARTS Print(...) WSM.

        Raises:
            Exception: If the variable has no associated workspace.
        """
        if (self.ws):
            self.ws.Print(self, 1)
        else:
            raise Exception("Can't print variable without associated ARTS workspace.")

    @staticmethod
    def get_group_id(value):
        """ This static method is used to determine how (and if) a given python variable can
        be mapped to a ARTS workspace variable group. The returned group id is required to
        add the variable to a workspace.

        Args:
            value(any): The python variable to map to the ARTS group.

        Returns:
            int: The index of the group which can be used to represent the python variable
                 or None if the type is not supported.
        """
        if type(value) == WorkspaceVariable:
            return group_ids[value.group]
        elif type(value) == Agenda:
            return group_ids["Agenda"]
        elif type(value) == int:
            return group_ids["Index"]
        elif type(value) == float or type(value) == np.float32 or type(value) == np.float64:
            return group_ids["Numeric"]
        elif type(value) == str:
            return group_ids["String"]
        elif type(value) == np.ndarray and value.ndim == 1:
            return group_ids["Vector"]
        elif type(value) == np.ndarray and value.ndim == 2:
            return group_ids["Matrix"]
        elif type(value) == np.ndarray and value.ndim == 3:
            return group_ids["Tensor3"]
        elif type(value) == np.ndarray and value.ndim == 4:
            return group_ids["Tensor4"]
        elif type(value) == np.ndarray and value.ndim == 5:
            return group_ids["Tensor5"]
        elif type(value) == np.ndarray and value.ndim == 6:
            return group_ids["Tensor6"]
        elif type(value) == list:
            group_name = ""
            nested_value = value
            while type(nested_value) == list and len(nested_value) > 0:
                nested_value = nested_value[0]
                group_name += "ArrayOf"
            if type(nested_value) == list and len(nested_value) == 0:
                raise ValueError("Empty lists are currently not handled.")
            else:
                t = type(nested_value)
                if t == str:
                    group_name += "String"
                    return group_ids[group_name]
                elif t == int:
                    group_name += "Index"
                    return group_ids[group_name]
                elif (t == np.float) or (t == np.float32) or (t == np.float64)  \
                     or (t == np.float128):
                    raise ValueError("Vectors, Matrices or Tensors should be"   \
                                     + " passed as numpy.ndarray and not as"    \
                                     " lists.")
                elif hasattr(nested_value, 'write_xml') \
                     and t.__name__ in group_names:

                    group_name += t.__name__
                    return group_ids[group_name]
                elif isinstance(nested_value, np.ndarray):
                    group_name += tensor_names[len(nested_value.shape) - 1]
                    return group_ids[group_name]
                else:
                    raise ValueError("Nested array with internal type " +       \
                                     str(t) + " not supported.")
        elif hasattr(value, 'write_xml') and type(value).__name__ in group_names:
            return group_ids[type(value).__name__]
        else:
            raise ValueError("Type " + str(type(value)) + " currently not supported.")

    @staticmethod
    def convert(group, value):
        """ Tries to convert a given python object to an object of the python class
        representing the given ARTS WSV group.

        Args:
            group(string): The name of an ARTS WSV group.
            group(any):    The object to convert

        Returns:
            (any): The converted object.
        """
        if (group == "Index"):
            return int(value)
        if (group == "String"):
            return value
        if (group == "ArrayOfString"):
            return [str(i) for i in value]
        if (group == "Numeric"):
            return np.float64(value)
        if (group == "Vector"):
            return np.array(value, dtype=np.float64, order='C', ndmin=1)
        if (group == "Matrix"):
            return np.array(value, dtype=np.float64, order='C', ndmin=2)
        return None

    @staticmethod
    def iter():
        """
        Iterator returning a WorkspaceVariable object for each ARTS WSV available.
        """
        for i in range(arts_api.get_number_of_variables()):
            s = arts_api.get_variable(i)
            name        = s.name.decode("utf8")
            description = s.description.decode("utf")
            group       = group_names[s.group]
            yield WorkspaceVariable(i, name, group, description)

    @property
    def value(self):
        """ Return the value of the variable in a given workspace.

        By default this function will check the value in the workspace associated
        with the variable of in the workspace object provided as argument to the
        function call. If the variable has an associated workspace the workspace
        provided as argument will be ignored.

        Returns:
            The value of the workspace variable represented by an object of
            the corresponding python types.

        Raises:
            Exception: If the type of the workspace variable is not supported
            by the interface.

        """
        if (self.ws):
            ws = self.ws
        if not ws:
            raise ValueError("WorkspaceVariable object need Workspace to determine value.")

        v = arts_api.get_variable_value(ws.ptr, self.ws_id, self.group_id)
        if not v.initialized:
            raise Exception("WorkspaceVariable " + self.name + " is uninitialized.")

        if self.group == "Index":
            return c.cast(v.ptr, c.POINTER(c.c_long))[0]
        elif self.group == "Numeric":
            return c.cast(v.ptr, c.POINTER(c.c_double))[0]
        elif self.group == "String":
            return (c.cast(v.ptr, c.c_char_p)).value.decode("utf8")
        elif self.group == "ArrayOfIndex":
            return [c.cast(v.ptr, c.POINTER(c.c_long))[i]
                    for i in range(v.dimensions[0])]
        elif self.group == "Sparse":
            m    = v.dimensions[0]
            n    = v.dimensions[1]
            nnz  = v.dimensions[2]
            if nnz == 0:
                return sp.sparse.csr_matrix(0)
            else:
                data = np.ctypeslib.as_array(c.cast(v.ptr,
                                                    c.POINTER(c.c_double)),
                                             (nnz,))
                row_indices = np.ctypeslib.as_array(v.inner_ptr, (nnz,))
                col_starts  = np.ctypeslib.as_array(v.outer_ptr, (m + 1,))
                return sp.sparse.csr_matrix((data, row_indices, col_starts),
                                            shape=(m,n))
        elif self.group == "Agenda":
            return Agenda(v.ptr)
        elif self.ndim:
            shape = []
            size  = 1
            for i in range(self.ndim):
                shape.append(v.dimensions[i])
                size *= v.dimensions[i]
            if size > 0:
                self.__array_interface__ = {"shape"  : tuple(shape),
                                            "typestr" : "|f8",
                                            "data" : (v.ptr, False),
                                            "version" : 3}
                return np.asarray(self)
            else:
                return np.zeros(shape)
        else:
            try:
                return self.to_typhon()
            except:
                raise Exception("Type of workspace variable is not supported "
                                + " by the interface.")

    def update(self):
        """ Update data references of the object.

        References to vector, matrices and tensors may change and must therefore
        be updated dynamically to ensure they are consistent with the state of
        the associated workspace. This method takes care of that.

        """
        if not self.ws==None and self.ndim:
            v = arts_api.get_variable_value(self.ws.ptr, self.ws_id, self.group_id)
            shape = []
            for i in range(self.ndim):
                shape.append(v.dimensions[i])
            self.__array_interface__ = {"shape"  : tuple(shape),
                                        "typestr" : "|f8",
                                        "data" : (v.ptr, False),
                                        "version" : 3}

    def erase(self):
        """
        Erase workspace variable from its associated workspace.
        """
        if self.ws:
            arts_api.erase_variable(self.ws.ptr, self.ws_id, self.group_id)
            self.ws = None

    def describe(self):
        """
        Print the description of the variable as given in ARTS methods.cc
        """
        print(self.description.format())

    def to_typhon(self):
        """
        Return the value of this variable as a typhon type. This function
        writes the value of the variable to a temporary file and reads it
        into Python using typhon load function. The purpose of this function
        is to access WSV whose groups are not natively supported by the
        C API.

        Returns:
            A typhon object with the same value as the WSV in the associated
            workspace.
        """
        if not self.ws:
            raise Exception("Cannot retrieve the value of a variable without "
                            + " associated Workspace.")
        tmp = tempfile.NamedTemporaryFile()
        self.ws.WriteXML("ascii", self, tmp.name)
        v = load(tmp.name)
        tmp.close()
        return v

    def from_typhon(self, var):
        """
        Set the value of this WSV in the associated workspace to the given
        typhon type. This function writes the value in ASCII format to a
        temporary file and reads it into the workspace

        Args:
            var: The value to which this WSV should be set in the associated
                 workspace.

        """
        if not self.ws:
            raise Exception("Cannot set the value of a variable without "
                            + " associated Workspace.")
        tmp = tempfile.NamedTemporaryFile()
        save(var, tmp.name)
        self.ws.ReadXML(self, tmp.name)


# Get ARTS WSV groups
group_names = [arts_api.get_group_name(i).decode("utf8")
               for i in range(arts_api.get_number_of_groups())]
group_ids = dict([(id, name) for (name,id) in enumerate(group_names)])


workspace_variables = dict()
for v in WorkspaceVariable.iter():
    globals()[v.name] = v
    workspace_variables[v.name] = v

