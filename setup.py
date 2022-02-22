from setuptools import setup
import sys
import os

__version__ = '0.0.1'

setup(
    name='PySurfaceOpt',
    long_description='',
    install_requires=['numpy', 'scipy', 'argparse', 'matplotlib', 'wheel', 'rich', 'bentley_ottmann', 'mpi4py'],
    packages = ["pysurfaceopt"],
    package_dir = {"pysurfaceopt": "pysurfaceopt"},
    package_data={'pysurfaceopt': ['data/*', 'data/ncsx/*']},
    include_package_data=True,
    zip_safe=False,
)
