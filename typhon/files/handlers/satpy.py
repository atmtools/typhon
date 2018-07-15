from .common import expects_file_info, FileHandler


class SatPy(FileHandler):
    """Wrapper for SatPy readers

    SatPy provides many readers for different satellite files. With this
    simple wrapper class, you can use them with typhon's FileSet without
    requiring any extra code!
    """

    def __init__(self, satpy_reader):
        super(SatPy, self).__init__()

        self.satpy_reader = satpy_reader

    @expects_file_info
    def read(self, filename, **kwargs):
        pass