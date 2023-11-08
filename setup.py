import os
import glob
from setuptools import Extension, setup


import numpy as np
from Cython.Build import cythonize

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


def find_extensions(src: str):
    return [
        Extension(
            os.path.splitext(os.path.basename(path))[0],
            [path],
            include_dirs=include_dirs,
            define_macros=define_macros,
        )
        for path in glob.glob(os.path.join(src, "*.pyx"))
    ]


extension_modules = find_extensions("fastspa")
extension_modules = [
    Extension(
        "fastspa._core",
        ["fastspa/_core.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
    ),
    Extension(
        "fastspa._lib",
        ["fastspa/_lib.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
    ),
    # Extension(
    #     "fastspa._view",
    #     ["fastspa/_view.pxd"],
    #     include_dirs=include_dirs,
    #     define_macros=define_macros,
    # ),
]

setup(
    ext_modules=cythonize(extension_modules, compiler_directives=compiler_directives),
)
