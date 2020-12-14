from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

version = "0.0.1a1"

ext_modules = [
    Extension(
        "dfit",
        ["src/dfit/dfit.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="dfit",
    version=version,
    author="Ashutosh Varma",
    license="GPL",
    ext_modules=cythonize(ext_modules),
    package_dir={"": "src/dfit"},
    install_requires=[
        "pandas>=1.1.4",
        "numpy>=1.19.4",
        "scipy>=1.5.4",
        "matplotlib>=3.3.3",
        "joblib>=0.17.0",
    ],
)
