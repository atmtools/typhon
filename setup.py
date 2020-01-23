"""Typhon is a collection of tools for atmospheric research.

Typhon provides:

- reading and writing routines for ARTS XML files
- an API to run and access ARTS through Python
- conversion routines for various physical quantities
- a tool kit to collocate different data sets (e.g. satellite, ship, ...)
- different retrievals (e.g. QRNN, SPARE-ICE, ...)
- a subset of the cmocean color maps
- functions to calculate and analyse cloud masks
- various plotting utility functions
- functions for geodetic and geographical calculations
- interface to the SRTM30 global elevation model
- functions for cloudmask statistics
- and much more...

Further information on ARTS can be found on http://www.radiativetransfer.org/.
"""

import logging
import subprocess

from setuptools import setup, find_packages
from codecs import open
from os.path import dirname, join

import builtins

builtins.__TYPHON_SETUP__ = True
DOCLINES = (__doc__ or "").split("\n")

version = open(join(dirname(__file__), "typhon", "VERSION")).read().strip()

if "dev" in version:
    try:
        cp = subprocess.run(
            ["git", "describe", "--tags"], stdout=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError:
        logging.warning(
            "Warning: could not determine version from git, "
            "using version from source"
        )
    else:
        so = cp.stdout
        version = (
            so.strip()
            .decode("ascii")
            .lstrip("v")
            .replace("-", "+dev", 1)
            .replace("-", ".")
        )

__version__ = version

setup(
    name="typhon",
    author="The Typhon developers",
    author_email="typhon.mi@lists.uni-hamburg.de",
    version=__version__,
    url="https://github.com/atmtools/typhon",
    download_url="https://github.com/atmtools/typhon/tarball/v" + __version__,
    packages=find_packages(),
    license="MIT",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    classifiers=[
        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires="~=3.6",
    include_package_data=True,
    install_requires=[
        "docutils",
        "imageio",
        "matplotlib>=1.4",
        "netCDF4>=1.1.1",
        "numba",
        "numexpr",
        "numpy>=1.13",
        "pandas",
        "scikit-image",
        "scikit-learn",
        "scipy>=0.15.1",
        "setuptools>=0.7.2",
        "xarray>=0.10.2",
    ],
    extras_require={
        "docs": ["cartopy", "pint", "sphinx_rtd_theme"],
        "tests": ["pytest", "pint", "gdal"],
    },
)
