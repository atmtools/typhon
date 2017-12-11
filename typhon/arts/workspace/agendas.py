"""The agendas submodule.

This module provides the Agenda class, which is used to represent parsed
controlfiles and can be executed on a given `Workspace` object.

"""

import ctypes as c
import numpy  as np
import os

from typhon.arts.workspace.api import find_controlfile, arts_api
from typhon.arts.workspace.output import CoutCapture


class Agenda:
    def __init__(self, ptr):
        """ Initialize Agenda object from pointer to C API Agenda object.
        Args:
            ptr(c.c_void_p): Pointer to Agenda object created with the parse_agenda
            method of the ARTS C API.
        """
        self.ptr = ptr

    def clear(self):
        arts_api.agenda_clear(self.ptr)

    def add_method(*args, **kwargs):
        if len(args) < 3:
            raise Exception("Need at least self, a workspace and the method to add as arguments.")
        self = args[0]
        ws   = args[1]
        m    = args[2]
        m_id, args_out, args_in, temps = m._parse_output_input_lists(ws, args[3:], kwargs)
        arg_out_ptr = c.cast((c.c_long * len(args_out))(*args_out), c.POINTER(c.c_long))
        arg_in_ptr = c.cast((c.c_long * len(args_in))(*args_in), c.POINTER(c.c_long))
        if not m.name[-3:] == "Set":
            for t in temps:
                arts_api.agenda_insert_set(ws.ptr, self.ptr, t.ws_id, t.group_id)
            arts_api.agenda_add_method(c.c_void_p(self.ptr), m_id,
                                       len(args_out), arg_out_ptr,
                                       len(args_in), arg_in_ptr)
        else:
            group_id = arts_api.get_variable(args_out[0]).group
            arts_api.agenda_insert_set(ws.ptr, self.ptr, args_out[0], group_id)

    def execute(self, ws):
        """ Execute this agenda on the given workspace.
        Args:
            ws(Workspace): Workspace object on wich to execute the agenda.
        Raises:
            Exception: If execution of agenda on workspace fails.
        """
        with CoutCapture(ws):
            e = arts_api.execute_agenda(ws.ptr, self.ptr)
        if (e):
            raise Exception("Error during execution of Agenda:\n" + e.decode("utf8"))

    def _to_value_struct(self):
        return {'ptr' : self.ptr}

    def __del__(self):
        """Destroys ARTS C API Agenda object associated with this Agenda object."""
        try:
            arts_api.destroy_agenda(self.ptr)
        except:
            pass

    @staticmethod
    def parse(name):
        """Parse controlfile and return agenda representing the agenda.
        Args:
            name(str): Name of the control file. Is looked up recursively in the path
            specified by the ARTS_INCLUDE_PATH environmental variable.
        Raises:
            Exception: If parsing of the controlfile fails.
        """
        path = find_controlfile(name)
        ptr = arts_api.parse_agenda(path.encode())
        if not ptr:
            e = arts_api.get_error().decode("utf8")
            raise Exception("Error during parsing of controlfile " + str(path) + ":\n" + e)
        return Agenda(ptr)
