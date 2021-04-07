from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="CDLoop", ext_modules=cythonize('CDLoop.pyx',annotate=True), include_dirs=[numpy.get_include()])
