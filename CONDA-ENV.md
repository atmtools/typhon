# Anaconda development environment
We strongly recommend to use [Anaconda](https://www.anaconda.com/) for your
Python development.

After downloading and installing Anaconda, you have to create a local 
working copy of the typhon repository:
```bash
$ git clone https://github.com/atmtools/typhon.git  # or your own fork
$ cd typhon
```
We compiled a [brief introduction](GITHUB-INTRO.md) on the general workflow
with GitHub.

On the command line, you can use the ``conda`` command to install required 
dependencies and libraries:
```bash
$ conda config --set channel_priority strict
$ conda config --append channels conda-forge
$ conda install --file requirements.txt
```

Finally, ``pip`` can be used to install the cloned working copy
to your Python environment (make sure to use the ``pip`` installed with 
``conda`` and not the system version.)
```bash
$ pip install --no-deps --user --editable .
```

This will install the package in editable mode (develop mode) in the user's
home directory. That way, local changes to the package are directly available
in the current environment.

With all the dependencies available you can now run the checks
```bash
pytest
```
and also build the documentation in the `doc/` subdirectory: 
```bash
cd doc
make clean html
```

The documentation is now available as HTML files in `_build/html/`.
