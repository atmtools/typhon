[![Build Status](https://travis-ci.org/atmtools/typhon.svg?branch=master)](https://travis-ci.org/atmtools/typhon)
![repo size](https://reposs.herokuapp.com/?path=atmtools/typhon)
[![Anaconda-Server Badge](https://anaconda.org/rttools/typhon/badges/installer/conda.svg)](https://anaconda.org/rttools/typhon)

# typhon - Tools for atmospheric research

## Requirements
Typhon requires Python version 3.6 or higher. The recommended way to get Python
is through [Anaconda]. But of course, any other Python distribution is also
working.

## Installation
The easiest way to develop typhon is to install the cloned working copy in your
Python environment. This can be done using ``pip``:
```bash
$ git clone https://github.com/atmtools/typhon.git
$ cd typhon
$ pip install --user --editable .
```

This will install the package in editable mode (develop mode) in the user's
home directory. That way, local changes to the package are directly available
in the current environment.

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

It is also possible to set environment variables in the same-named
section of the configuration file, e.g.:
```
[environment]
ARTS_BUILD_PATH: /path/to/arts/build/
```

## Documentation
The documentation of the project is created with [Sphinx]. You can use the
enclose Makefile to build your own documentation:
```bash
$ cd doc
$ make html
```

A daily build version of the documentation is accessible
[online](http://radiativetransfer.org/misc/typhon/doc-trunk).
Kindly note that bleeding edge features might not be covered.

[Sphinx]: http://www.sphinx-doc.org
[Anaconda]: https://www.continuum.io/downloads
[pytest]: https://docs.pytest.org/
