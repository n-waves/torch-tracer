# cython: language_level=3
import sys

cdef extern from "<time.h>" nogil:
    ctypedef long time_t
    ctypedef int clockid_t
    cdef enum:
        CLOCK_REALTIME = 0
    cdef struct timespec:
        time_t tv_sec
        long tv_nsec
    int clock_gettime(clockid_t, timespec *)
from cpython.ref cimport PyObject, Py_INCREF, Py_XDECREF

cdef extern from "code.h":
    ctypedef struct PyCodeObject:
        PyObject *co_filename
        PyObject *co_name

cdef extern from "frameobject.h":
    ctypedef struct PyFrameObject:
        PyFrameObject *f_back
        PyCodeObject *f_code
        int f_lineno

from cpython.pystate cimport (
    Py_tracefunc,
    PyTrace_CALL, PyTrace_EXCEPTION, PyTrace_LINE, PyTrace_RETURN,
    PyTrace_C_CALL, PyTrace_C_EXCEPTION, PyTrace_C_RETURN)

cdef extern from *:
    void PyEval_SetProfile(Py_tracefunc cfunc, PyObject *obj)

cdef int profile(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg) except -1:
    cdef timespec ts
    cdef list target = <list>obj

    clock_gettime(CLOCK_REALTIME, &ts)
    timestamp = ts.tv_sec * 1000000000LL + ts.tv_nsec

    cdef object code = <object>frame.f_code
    #cdef object back = <object>frame.f_back
    target.append((code.co_filename, code.co_name, frame.f_lineno, what, timestamp))
    return 0

def set_recorder(recorder):
    if recorder is not None:
        PyEval_SetProfile(<Py_tracefunc>profile, <PyObject*>recorder._log)
    else:
        PyEval_SetProfile(NULL, NULL)
    return None

class Dummy:
  pass

def setstatprofile(target, interval=0.001):
    a = Dummy()
    a._log = []
    set_recorder(a)
