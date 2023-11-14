import sys
import os
import glob
from setuptools import Extension, setup


import numpy as np
from Cython.Build import cythonize


compiler_directives: dict[str, int | bool] = {"language_level": 3}
define_macros: list[tuple[str, str | None]] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if '--coverage' in sys.argv:
    sys.argv.remove('--coverage')
    # in order to compile the cython code for test coverage
    # we need to include the following compiler directives...
    compiler_directives.update({"linetrace": True, "profile": True})
    # and include the following trace macros
    define_macros.extend([("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")])

include_dirs = [".", np.get_include()]

def find_extensions(src: str, p: str, **kwargs):
    return [
        Extension(
            f"{src}.{os.path.splitext(os.path.basename(m))[0]}",
            [os.path.join(src, m)],
            **kwargs,
        )
        for m in glob.glob(p, root_dir=src)
    ]


extension_modules = find_extensions("fast_spa", "*.pyx", include_dirs=include_dirs, define_macros=define_macros)

setup(
    ext_modules=cythonize(extension_modules, compiler_directives=compiler_directives),
)
