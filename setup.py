import os
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

# os.environ["TEST"] = "TRUE"
TEST = os.environ.get("TEST") == "TRUE"


compiler_directives: dict[str, int | bool] = {"language_level": 3}
define_macros: list[tuple[str, str | None]] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if TEST:
    # in order to compile the cython code for test coverage
    # we need to include the following compiler directives...
    compiler_directives.update({"linetrace": True, "profile": True})
    # and include the following trace macros
    define_macros.extend([("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")])


extension_modules = [
    Extension(
        "fastspa.lib",
        ["fastspa/lib.pyx", "fastspa/tab.pyx"],
        include_dirs=[np.get_include(), "fastspa/"],
        define_macros=define_macros,
    ),
]


setup(
    name="fastspa",
    ext_modules=cythonize(extension_modules, compiler_directives=compiler_directives),
)
