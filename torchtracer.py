import sys, os
import recorder_cython
from pyinstrument.vendor.six import exec_, PY2

class Recorder:
  def __init__(self):
    self._log = []

  def start(self):
    recorder_cython.set_recorder(self)

  def stop(self):
    recorder_cython.set_recorder(None)

  def __enter__(self):
    self.start()

  def __exit__(self, *args):
    self.stop()

PyTrace_CALL = 0
PyTrace_EXCEPTION = 1
PyTrace_LINE = 2
PyTrace_RETURN = 3
PyTrace_C_CALL = 4
PyTrace_C_EXCEPTION = 5
PyTrace_C_RETURN = 6

trace_types = {
    PyTrace_CALL:        'call',
    PyTrace_EXCEPTION:   'exception',
    PyTrace_LINE:        'line',
    PyTrace_RETURN:      'return',
    PyTrace_C_CALL:      'ccall',
    PyTrace_C_EXCEPTION: 'cexc',
    PyTrace_C_RETURN:    'cret',
}

def dump_record(log, filename):
  import pandas as pd
  import sqlite3

  def rank_dict(l):
    d = {k:i for i, k in enumerate(set(l))}
    l = [d[k] for k in l]
    d = {v: k for k, v in d.items()}
    d = pd.Series(d, name='value')
    return d, l

  filenames, functions, lines, whats, timestamps = zip(*log)
  print("1")
  dfiles, filenames = rank_dict(filenames)
  print("2")
  dfuncs, functions = rank_dict(functions)
  print("3")
  dtypes = pd.Series(trace_types, name='value')
  print("4")

  records = pd.DataFrame({
    'filename': filenames,
    'function': functions,
    'line': lines,
    'what': whats,
    'timestamp': timestamps})

  with sqlite3.connect(filename) as conn:
    print("5")
    opts = {'if_exists': 'replace', 'index_label': 'id'}
    print("6")
    dfiles.to_sql('FILENAMES', conn, **opts)
    print("7")
    dfuncs.to_sql('FUNCTIONS', conn, **opts)
    print("8")
    dtypes.to_sql('TRACE_TYPES', conn, **opts)
    print("9")
    records.to_sql('RECORDS', conn, **opts)
    print("0")


def main():
  recorder = Recorder()
  sys.argv[:1] = []
  progname = sys.argv[0]
  filename = 'cpu.db'
  sys.path.insert(0, os.path.dirname(progname))
  with open(progname, 'rb') as fp:
      code = compile(fp.read(), progname, 'exec')
  globs = {
      '__file__': progname,
      '__name__': '__main__',
      '__package__': None,
  }

  with recorder:
    try:
      exec_(code, globs, None)
    except (SystemExit, KeyboardInterrupt):
       pass

  dump_record(recorder._log, filename)

if __name__ == '__main__':
  main()
