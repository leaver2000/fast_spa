import os
import glob
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

os.environ["TEST"] = "TRUE"
TEST = os.environ.get("TEST") == "TRUE"


compiler_directives: dict[str, int | bool] = {"language_level": 3}
define_macros: list[tuple[str, str | None]] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if TEST:
    # in order to compile the cython code for test coverage
    # we need to include the following compiler directives...
    compiler_directives.update({"linetrace": True, "profile": True})
    # and include the following trace macros
    define_macros.extend([("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")])

include_dirs = [".", np.get_include()]
extension_modules = cythonize(
    [
        Extension(
            "fastspa.landscaping",
            ["fastspa/landscaping.pyx"],
            include_dirs=include_dirs,
            define_macros=define_macros,
        ),
        Extension(
            "fastspa.shrubbing",
            ["fastspa/shrubbing.pyx"],
            include_dirs=include_dirs,
            define_macros=define_macros,
        ),
    ]
)

# from setuptools import setup
# from Cython.Build import cythonize
# extension_modules = cythonize("fastspa/*.pyx")
# print(ext_modules[0])
setup(ext_modules=extension_modules, define_macros=define_macros)
#     ext_modules=cythonize("fastspa/*.pyx", define_macros=define_macros),
#     # package_data={"fastspa": ["*.pxd"]},
# )
