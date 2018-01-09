from tempfile import gettempdir, NamedTemporaryFile

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
        with NamedTemporaryFile(dir=gettempdir()) as file:
            with compress(file.name+".zip") as compressed_file:
                self.create_file(compressed_file)

            with decompress(file.name+".zip") as uncompressed_file:
                assert self.check_file(uncompressed_file)

    def test_compress_decompress_gzip(self):
        with NamedTemporaryFile(dir=gettempdir()) as file:
            with compress(file.name+".gz") as compressed_file:
                self.create_file(compressed_file)

            with decompress(file.name+".gz") as uncompressed_file:
                assert self.check_file(uncompressed_file)

    def test_compress_decompress_bz2(self):
        with NamedTemporaryFile(dir=gettempdir()) as file:
            with compress(file.name+".bz2") as compressed_file:
                self.create_file(compressed_file)

            with decompress(file.name+".bz2") as uncompressed_file:
                assert self.check_file(uncompressed_file)

    def test_compress_decompress_lzma(self):
        with NamedTemporaryFile(dir=gettempdir()) as file:
            with compress(file.name+".xz") as compressed_file:
                self.create_file(compressed_file)

            with decompress(file.name+".xz") as uncompressed_file:
                assert self.check_file(uncompressed_file)