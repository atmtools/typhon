"""Typhon is a collection of tools for atmospheric research.

It provides:
- reading and writing routines for ARTS XML files
- an API to run and access ARTS through Python
- conversion routines for various physical quantities
- a tool kit to collocate different data sets (e.g. satellite, ship, ...)
- different retrievals (e.g. QRNN, SPARE-ICE, ...)
- a subset of the cmocean color maps
- functions to calculate and analyse cloud masks
- various plotting utility functions
- functions for geodetic and geographical calculations
- and much more...

Further information on ARTS can be found on http://www.radiativetransfer.org/.
"""

import logging
import sys
import subprocess

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os.path import (abspath, dirname, join)

import builtins

builtins.__TYPHON_SETUP__ = True
DOCLINES = (__doc__ or '').split("\n")

version = open(join(dirname(__file__), 'typhon', 'VERSION')).read().strip()

if 'dev' in version:
    try:
        cp = subprocess.run(
            ["git", "describe", "--tags"],
            stdout=subprocess.PIPE,
            check=True)
        so = cp.stdout
        __version__ = so.strip().decode("ascii").lstrip("v").replace(
            "-", "+dev", 1).replace("-", ".")
    except subprocess.CalledProcessError:
        logging.warning(
            "Warning: could not determine version from git, extracting "
            " latest release version from source")

        # Parse version number from module-level ASCII file. This prevents
        # double-entry bookkeeping).
        __version__ = version
else:
    __version__ = version

here = abspath(dirname(__file__))

setup(
    name='typhon',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),

    # The project's main homepage.
    url='https://github.com/atmtools/typhon',

    # Author details
    author='The Typhon developers',
    author_email='typhon.mi@lists.uni-hamburg.de',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # What does your project relate to?
    # keywords='science radiative transfer',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'doc', 'tests*']),

    # Python requirements
    python_requires='>=3.6',

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'docutils',
        'imageio',
        'matplotlib>=1.4',
        'netCDF4>=1.1.1',
        'numba',
        'numexpr',
        'numpy>=1.13',
        'pandas',
        'scikit-image',
        'scikit-learn',
        'scipy>=0.15.1',
        'setuptools>=0.7.2',
        'xarray>=0.10.2',
    ],

    # Dependencies for additional features of the project (e.g. documentation).
    # You can install these using the following syntax:
    # >>> pip install .[docs]
    # or for multiple features:
    # >>> pip install .[docs,tests]
    extras_require={
        'docs': [
            'cartopy',
            'pint',
            'sphinx_rtd_theme',
        ],
        'tests': [
            'pytest',
            'pint',
            'gdal',
        ],
    },

    # Additional requirements to make `$ python setup.py test` work.
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #    'sample': ['package_data.dat'],
    # },

    include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    # },
)
