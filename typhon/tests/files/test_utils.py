import os
from tempfile import TemporaryDirectory

from typhon.files import compress, decompress


class TestCompression:
    data = "ABCDEFGHIJKLMNOPQRSTUVWXYZ12345678910"

    def create_file(self, filename):
        with open(filename, "w") as file:
            file.write(self.data)

    def check_file(self, filename):
        with open(filename) as file:
            return self.data == file.readline()

    def test_compress_decompress_zip(self):
        with TemporaryDirectory() as tdir:
            tfile = os.path.join(tdir, 'testfile')
            with compress(tfile + ".zip") as compressed_file:
                self.create_file(compressed_file)

            with decompress(tfile + ".zip") as uncompressed_file:
                assert self.check_file(uncompressed_file)

    def test_compress_decompress_gzip(self):
        with TemporaryDirectory() as tdir:
            tfile = os.path.join(tdir, 'testfile')
            with compress(tfile + ".gz") as compressed_file:
                self.create_file(compressed_file)

            with decompress(tfile + ".gz") as uncompressed_file:
                assert self.check_file(uncompressed_file)

    def test_compress_decompress_bz2(self):
        with TemporaryDirectory() as tdir:
            tfile = os.path.join(tdir, 'testfile')
            with compress(tfile + ".bz2") as compressed_file:
                self.create_file(compressed_file)

            with decompress(tfile + ".bz2") as uncompressed_file:
                assert self.check_file(uncompressed_file)

    def test_compress_decompress_lzma(self):
        with TemporaryDirectory() as tdir:
            tfile = os.path.join(tdir, 'testfile')
            with compress(tfile + ".xz") as compressed_file:
                self.create_file(compressed_file)

            with decompress(tfile + ".xz") as uncompressed_file:
                assert self.check_file(uncompressed_file)
