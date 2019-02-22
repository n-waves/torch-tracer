from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("recorder_cython", ["recorder.pyx"],
        include_dirs=["."])
]

setup(
    name = "recorder_cython",
    version = "0.1",
    ext_modules = cythonize(extensions)
)
