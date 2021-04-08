from distutils.core import setup
from Cython.Build import cythonize
import numpy
from setuptools import setup, find_packages, Extension

ext_modules=[ Extension("CDLoop",    # location of the resulting .so
             ["CDLoop.pyx"],) ]

# setup(name="nonlinear_causal", 
# 	ext_modules=cythonize('CDLoop.pyx',annotate=True), 
# 	include_dirs=[numpy.get_include()])

setup(name="nonlinear_causal", 
	ext_modules=ext_modules, 
	include_dirs=[numpy.get_include()])