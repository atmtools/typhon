from typhon.arts.workspace.api import find_controlfile, arts_api

"""ARTS Agendas

This module provides the Agenda class, which is used to represent parsed
controlfiles and can be executed on a given Workspace object.

"""
class Agenda:
    def __init__(self, ptr):
        """ Initialize Agenda object from pointer to C API Agenda object.
        Args:
            ptr(c.c_void_p): Pointer to Agenda object created with the parse_agenda
            method of the ARTS C API.
        """
        self.ptr = ptr

    def execute(self, ws):
        """ Execute this agenda on the given workspace.
        Args:
            ws(Workspace): Workspace object on wich to execute the agenda.
        Raises:
            Exception: If execution of agenda on workspace fails.
        """
        e = arts_api.execute_agenda(ws.ptr, self.ptr)
        if (e):
            raise Exception("Error during execution of Agenda:\n" + e)

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
