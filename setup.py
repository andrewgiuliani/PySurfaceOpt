from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import setuptools

__version__ = '0.0.1'

setup(
    name='PySurfaceOpt',
    long_description='',
    install_requires=['numpy', 'scipy', 'argparse', 'matplotlib', 'mayavi'],
    packages = ["pysurfaceopt"],
    package_dir = {"pysurfaceopt": "pysurfaceopt"},
    package_data={'pysurfaceopt': ['data/*', 'data/ncsx/*']},
    include_package_data=True,
    zip_safe=False,
)
