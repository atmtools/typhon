"""ARTS C API Interface

This module provides a foreign function interface for the ARTS C API.
It defines the C structs used by the interface as ctypes.Structure
child classes as well as the return argument and return types of the
function provided by the C API.

Requirements
------------

The ARTS C API is provided by the arts_api.so library and is required by
the module. The module will check if the ARTS_BUILD_PATH variable is set
and assume the library can be found in the src subdirectory. If opening
the library fails loading the module will fail with an EnvironmentError.

Attributes:

    arts_api(CDLL): The ctypes library handle holding the ARTS C API.

"""

import ctypes as c
import numpy  as np
import os

################################################################################
# Load ARTS C API
################################################################################

arts_build_path = os.environ.get('ARTS_BUILD_PATH')
if arts_build_path is None:
    raise EnvironmentError("ARTS_BUILD_PATH environment variable required to locate ARTS API.")

try:
    print ("Loading ARTS API from: " + arts_build_path + "/src/arts_api.so")
    arts_api = c.cdll.LoadLibrary(arts_build_path + "/src/libarts_api.so")
except:
    raise EnvironmentError("Could not find ARTS API in your ARTS build path. Did you install it?")

arts_api.initialize()

################################################################################
# Setup ARTS Environment
################################################################################
# Get ARTS include path.

arts_include_path = []
try:
    include_path = os.environ['ARTS_INCLUDE_PATH'] + "/controlfiles"
    data_path    = os.environ['ARTS_INCLUDE_PATH'] + "/controlfiles/testdata"
    arts_include_path = [include_path]
except:
    pass
arts_include_path.append(os.getcwd())

# Set runtime parameters
arts_api.set_parameters(c.c_char_p(include_path.encode()),
                        c.c_char_p(data_path.encode()))

def find_controlfile(name):
    """ Recursively search arts include path for given file.
    Args:
        name(str): Name of the file.
    Raises:
        Exception: If the file cannot be found.
    """
    path = None
    for p in arts_include_path:
        for root, dirs, files in os.walk(p):
            try:
                file = files[files.index(name)]
                path = root + "/" + file
                break
            except:
                pass
    if (path):
        return path
    else:
        raise Exception("File " + name + " not found. Search path was:\n "
                        + str(arts_include_path))
    return path

def include_path_add(path):
    """Add path to include path of the ARTS runtime.

    Args:
        path(str): Path to add to the ARTS include path.
    """
    arts_api.set_parameters(c.c_char_p(name.encode()),
                            c.c_char_p(None))

def data_path_add(name):
    """ Add path to data path of the ARTS runtime.

    Args:
        path(str): Path to add to the ARTS data path.
    """
    arts_api.set_parameters(c.c_char_p(None),
                            c.c_char_p(name.encode()))


################################################################################
# ctypes Structures
################################################################################

class VariableStruct(c.Structure):
    """
    A (symbolic) ARTS workspace variable is represented using a struct containing
    pointers to the name and description of the method as well as the group id,
    i.e. the Index variable encoding the type of the variable.
    """
    _fields_ = [("name", c.c_char_p),
                ("description", c.c_char_p),
                ("group", c.c_long)]


class VariableValueStruct(c.Structure):
    """
    The value of an instance of a workspace variable is represented by a struct
    containing a pointer to the relevant data, initialized flag, and an array
    of up to six dimension values. The dimension array is used by tensor data
    to transfer the dimensions of the object.
    """
    _fields_ = [("ptr", c.c_void_p),
                ("initialized", c.c_bool),
                ("dimensions", 6 * c.c_long)]

class MethodStruct(c.Structure):
    """
    The method struct holds the internal index of the method (id), pointers
    to the null-terminated strings holding name and description, the number
    of generic inputs (n_g_in) and a pointer to the array holding the group ids
    of the output types, as well as the number of generic outputs and their types.
    """
    _fields_ = [("id", c.c_ulong),
                ("name", c.c_char_p),
                ("description", c.c_char_p),
                # Output
                ("n_out", c.c_ulong),
                ("outs", c.POINTER(c.c_long)),
                # Generic Output
                ("n_g_out", c.c_ulong),
                ("g_out_types", c.POINTER(c.c_long)),
                # Input
                ("n_in", c.c_ulong),
                ("ins", c.POINTER(c.c_long)),
                # Generic Input
                ("n_g_in", c.c_ulong),
                ("g_in_types", c.POINTER(c.c_long))]

# TODO: Check if can be used as constructor
def variable_value_factory(value):
    """ Create a VariableValue struct from a python object.

    This functions creates a variable value struct from a python object so that it
    can be passed to the C API. If the type of the object is not supported, the data
    pointer will be NULL.

    Args:
        value(object): The python object to represent as a VariableValue struct.

    TODO: Add proper error handling.
    """
    ptr = 0
    initialized = True
    dimensions  = [0] * 6

    # Index
    if type(value) == int:
        ptr = c.cast(c.pointer(c.c_long(value)), c.c_void_p)
    # Numeric
    if type(value) == float or type(value) == np.float32 or type(value) == np.float64:
        temp = np.float64(value)
        ptr = c.cast(c.pointer(c.c_double(temp)), c.c_void_p)
    # String
    elif type(value) == str:
        ptr = c.cast(c.c_char_p(value.encode()), c.c_void_p)
    # Vector, Matrix
    elif type(value) == np.ndarray:
        if value.dtype == np.float64:
            ptr = value.ctypes.data
            for i in range(value.ndim):
                dimensions[i] = value.shape[i]
    # Array of String or Integer
    elif type(value) == list:
        if not value:
            raise ValueError("Empty lists currently not supported.")
        t = type(value[0])
        ps = []
        if t ==str:
            for s in value:
                ps.append(c.cast(c.c_char_p(s.encode()), c.c_void_p))
        if t == int:
            for i in value:
                ps.append(c.cast(c.pointer(c.c_long(i)), c.c_void_p))
        p_array = (c.c_void_p * len(ps))(*ps)
        ptr = c.cast(c.pointer(p_array), c.c_void_p)
        dimensions[0] = len(p_array)

    dimensions = (c.c_long * 6)(*dimensions)
    return VariableValueStruct(ptr, initialized, dimensions)

################################################################################
# Function Arguments and Return Types
################################################################################

# Create ArtsWorkspace and return handle.
arts_api.create_workspace.argtypes = None
arts_api.create_workspace.restype  = c.c_void_p

# Destroy ArtsWorkspace instance from handle.
arts_api.destroy_workspace.argtypes = [c.c_void_p]
arts_api.destroy_workspace.restype   = None

# Set include ad data path of the arts runtime.
arts_api.set_parameters.restype  = None
arts_api.set_parameters.argtypes = [c.c_char_p, c.c_char_p]

# Set include ad data path of the arts runtime.
arts_api.get_error.restype  = c.c_char_p
arts_api.get_error.argtypes = None

# Agendas
#
#
arts_api.parse_agenda.argtypes = [c.c_char_p]
arts_api.parse_agenda.restype  = c.c_void_p

arts_api.execute_agenda.argtypes = [c.c_void_p, c.c_void_p]
arts_api.execute_agenda.restype  = c.c_char_p

# Groups
#
# Returns the number of WSV groups.
arts_api.get_number_of_groups.argtypes = None
arts_api.get_number_of_groups.restype  = c.c_ulong

# Return pointer to the name of the group with given index.
arts_api.get_group_name.argtypes = [c.c_long]
arts_api.get_group_name.restype  = c.c_char_p

# Variables
#
# Returns the number of (symbolic) workspace variable.
arts_api.get_number_of_variables.restype  = c.c_ulong
arts_api.get_number_of_variables.argtypes = None

# Returns workspace variable with index c_long as VariableStruct.
arts_api.lookup_workspace_variable.argtypes = [c.c_char_p]
arts_api.lookup_workspace_variable.restype  = c.c_long

# Returns workspace variable with index c_long as VariableStruct.
arts_api.get_variable.argtypes = [c.c_long]
arts_api.get_variable.restype  = VariableStruct

# Return pointer to variable value in a given workspace in the form of a VariableValueStruct.
arts_api.get_variable_value.argtypes = [c.c_void_p, c.c_long, c.c_long]
arts_api.get_variable_value.restype  = VariableValueStruct

# Set variable value in workspace given a workspace handle, the variable id, the group id
# and a VariableValueStruct
arts_api.set_variable_value.argtypes = [c.c_void_p, c.c_long, c.c_long, VariableValueStruct]
arts_api.set_variable_value.restype  = None

# Adds a value of a given group to a given workspace.
arts_api.add_variable.restype  = c.c_long
arts_api.add_variable.argtypes = [c.c_void_p, c.c_long]

# Remove given variable from workspace.
arts_api.erase_variable.restype  = None
arts_api.erase_variable.argtypes = [c.c_void_p, c.c_long, c.c_long]

# Methods
#
# Returns the number of (symbolic) workspace variable.
arts_api.get_number_of_methods.restype  = c.c_ulong
arts_api.get_number_of_methods.argtypes = None

# Returns workspace variable with index c_long as VariableStruct.
arts_api.get_method.argtypes = [c.c_long]
arts_api.get_method.restype  = MethodStruct

# Return Pointer to name of jth generic output parameter of a given WSM.
arts_api.get_method_g_out.argtypes = [c.c_long, c.c_long]
arts_api.get_method_g_out.restype  = c.c_char_p

# Return Pointer to name of jth generic input parameter of a given WSM.
arts_api.get_method_g_in.argtypes = [c.c_long, c.c_long]
arts_api.get_method_g_in.restype  = c.c_char_p

# Return pointer to the default value of the jth generic input of a given WSM.
arts_api.get_method_g_in_default.argtypes = [c.c_long, c.c_long]
arts_api.get_method_g_in_default.restype  = c.c_char_p

# Execute a given workspace method.
arts_api.execute_workspace_method.restype  = c.c_char_p
arts_api.execute_workspace_method.argtypes = [c.c_void_p,
                                              c.c_long,
                                              c.POINTER(c.c_long),
                                              c.POINTER(c.c_long)]
