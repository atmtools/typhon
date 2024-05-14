[![PyPI version](https://badge.fury.io/py/typhon.svg)](https://badge.fury.io/py/typhon)
[![Anaconda-Server Badge](https://anaconda.org/rttools/typhon/badges/installer/conda.svg)](https://anaconda.org/rttools/typhon)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1300318.svg)](https://doi.org/10.5281/zenodo.1300318)
[![Test](https://github.com/atmtools/typhon/workflows/Test/badge.svg?branch=master)](https://github.com/atmtools/typhon/commits/master)

# typhon - Tools for atmospheric research

## Installation
Typhon requires Python version 3.10 or higher. The recommended way to get Python
is through [Miniforge3]. But of course, any other Python distribution is also
working.

### Stable release
The latest stable release of typhon can be installed using ``conda`` 
(recommended)
```bash
$ conda install -c rttools typhon
```
or ``pip``
```bash
$ pip install typhon
```

### Development version
Check our information on how to [setup a development environment](CONDA-ENV.md)
for typhon.

## Testing
Typhon contains a simple testing framework using [pytest]. It is good
practice to write tests for all your functions and classes. Those tests may not
be too extensive but should cover the basic use cases to ensure correct
behavior through further development of the package.

Tests can be run on the command line...
```bash
$ pytest --pyargs typhon
```
or using the Python interpreter:
```python
import typhon
typhon.test()
```

## Configuration
Typhon supports a configuration file in ``configparser`` syntax. The
configuration is handled by the ``typhon.config`` module. The default file
location is ``~/.typhonrc`` but can be changed using the ``TYPHONRC``
environment variable.

## Documentation
A recent build of the documentation is accessible
[online](http://radiativetransfer.org/misc/typhon/doc-trunk).
Kindly note that bleeding edge features might not be covered.

[Sphinx]: http://www.sphinx-doc.org
[Miniforge3]: https://github.com/conda-forge/miniforge#miniforge
[pytest]: https://docs.pytest.org/
