#!/usr/bin/env python
import builtins
import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

from distutils.errors import DistutilsOptionError

import versioneer

try:
    from Cython.Build import cythonize

    CYTHON_INSTALLED = True
except ImportError:
    CYTHON_INSTALLED = False


COMPILED_MODULES = [
    "dfit.dfit",
]
SOURCE_PATH = Path("src")


## OPTION HANDLING


def has_option(name):
    try:
        sys.argv.remove(f"--{name}")
        return True
    except ValueError:
        pass
    # allow passing all cmd line options also as environment variables
    env_val = os.getenv(name.upper().replace("-", "_"), "false").lower()
    if env_val == "true":
        return True
    return False


def option_value(name):
    for index, option in enumerate(sys.argv):
        if option == f"--{name}":
            if index + 1 >= len(sys.argv):
                raise DistutilsOptionError("The option %s requires a value" % option)
            value = sys.argv[index + 1]
            sys.argv[index : index + 2] = []
            return value
        if option.startswith(f"--{name}="):
            value = option[len(name) + 3 :]
            sys.argv[index : index + 1] = []
            return value
    env_name = name.upper().replace("-", "_")
    env_val = os.getenv(env_name)
    return env_val


OPTION_WITHOUT_CYTHON = has_option("without-cython") or not CYTHON_INSTALLED
OPTION_WITH_CYTHON_GDB = has_option("cython-gdb")
OPTION_WITH_REFNANNY = has_option("with-refnanny")

# enable these three for profiling and showing function signature
# in sphnix
OPTION_WITH_COVERAGE = has_option("with-coverage")
OPTION_WITH_CLINES = has_option("with-clines")
OPTION_WITH_SIGNATURE = has_option("with-signature")


## EXTENSION BUILDING


def define_macros():
    macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    if OPTION_WITH_REFNANNY:
        macros.append(("CYTHON_REFNANNY", None))
    if OPTION_WITH_COVERAGE:
        macros.append(("CYTHON_TRACE_NOGIL", "1"))
    # Disable showing C lines in tracebacks, unless explicitly requested.
    macros.append(("CYTHON_CLINE_IN_TRACEBACK", "1" if OPTION_WITH_CLINES else "0"))
    return macros


def get_ext_modules():
    modules = {
        m: SOURCE_PATH
        / Path(
            *m.split(".")[:-1],
            m.split(".")[-1] + (".cpp" if OPTION_WITHOUT_CYTHON else ".pyx"),
        )
        for m in COMPILED_MODULES
    }

    cythonize_directives = {}
    gdb_debug = True if OPTION_WITH_CYTHON_GDB else False

    if OPTION_WITH_COVERAGE:
        cythonize_directives["linetrace"] = True
    if OPTION_WITH_SIGNATURE:
        cythonize_directives["embedsignature"] = True

    result = []
    for m, src in modules.items():
        if not src.is_file():
            raise RuntimeError(
                f"ERROR: Trying to build but '{src}'" " is not available"
            )
        result.append(
            Extension(
                m,
                sources=[
                    str(src),
                ],
                define_macros=define_macros(),
            )
        )

    if not OPTION_WITHOUT_CYTHON:
        return cythonize(
            result, compiler_directives=cythonize_directives, gdb_debug=gdb_debug
        )
    else:
        return result


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        builtins.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())

cmdclass = versioneer.get_cmdclass()
cmdclass.update(build_ext=build_ext)

# Give setuptools a hint to complain if it's too old a version
# 30.3.0 allows us to put most metadata in setup.cfg
# Should match pyproject.toml & setup.cfg
SETUP_REQUIRES = ["setuptools >= 38.3.0"]
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ["wheel"] if "bdist_wheel" in sys.argv else []

if __name__ == "__main__":
    if CYTHON_INSTALLED and not OPTION_WITHOUT_CYTHON:
        print("Using cython to build cpp sources")
    elif OPTION_WITHOUT_CYTHON:
        print("Using prebuilt cpp sources")

    setup(
        ext_modules=get_ext_modules(),
        setup_requires=SETUP_REQUIRES,
        cmdclass=cmdclass,
    )
