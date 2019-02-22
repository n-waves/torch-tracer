import sys
import sqlite3
import pandas as pd
from pyinstrument.frame import Frame, SelfTimeFrame
from pyinstrument.renderers import ConsoleRenderer
from pyinstrument import processors
import numba
import re
from collections import Counter
import fire
import json

  
cuda_seq_re = re.compile(r'^ ((stashed seq=(?P<stashed>\d+))|(seq=(?P<seq>\d+)))$')
class MyEncoder(json.JSONEncoder):
  def default(self, o):
    return o.__dict__

class AggregateFrame:
  def __init__(self, identifier):
    self.identifier = identifier
    self.self_time = 0.0
    self.children = {}
    self.seq = None
    self.function = None
    

def from_json(obj):
  f = AggregateFrame(obj['identifier'])
  f.self_time = obj['self_time']
  f.children = {k:from_json(ch) for k, ch in obj['children'].items()}
  f.seq = obj.get('seq')
  f.function = obj.get('function')

  return f
#  def add_self_time(self, time):
#    self.self_time += time
#
#  def add_child(self, frame):
#    if frame.identifier in self.children:
#      return self.children[frame.identifier]
#    self.children[frame.identifier] = frame
#    return frame


def to_frame(agg):
  if agg.seq is None:
    identifier = agg.identifier
  else:
    function, filename, line = agg.identifier.split('\x00')
    identifier = f"{function}:{agg.seq}\x00{filename}\x00{line}"
  frame = Frame(identifier)
  frame.add_children([to_frame(ch) for ch in agg.children.values()])
  frame.add_child(SelfTimeFrame(self_time=agg.self_time))
  return frame


@numba.jit()
def aggregate(cuda):
  print("aggregating...")
  root = AggregateFrame('<root>\x00file.py\x000')
  stack = [root]
  seq_register = {}
  op_counter = Counter()

  for index, row in cuda.iterrows():
    stashed, seq = None, None
    if len(stack) == 0:
      print("stack is empty")
      stack = [root]
    ts, filename, function, line, tpe = row
    if filename == '<cuda>':
      parts = function.split(',')
      function = parts[0]
      if len(parts) > 1:
        m = cuda_seq_re.match(parts[1])
        if m:
          stashed, seq = m['stashed'], m['seq']

    name = str(function) + '\x00' + str(filename) + '\x00' + str(line)
    ts /= 1000000000.0
  
    parent = stack[-1]
    parent.self_time += ts
    if tpe == 'start':
      if stashed:
        if stashed in seq_register:
          name = str(function) + ' (' + str(seq_register[stashed].function) + ':' + str(seq_register[stashed].seq) + ')\x00' + str(filename) + '\x00' + str(line)
        else:
          pass #print("Stashed operation " + str(function) + " stashed seq=" + str(stashed) + " not found")
      if name in parent.children:
        frame = parent.children[name]
      else:
        frame = AggregateFrame(name)
        parent.children[name] = frame
        if seq:
          frame.seq = op_counter[function]
          op_counter.update([function])
      stack.append(frame)
      if seq:
        seq_register[seq] = frame
        frame.function = function
    else:
      stack.pop()
  return root


def transform_cpu(cpu):
  rem = cpu['filename'].fillna('<nan>').str.endswith('torchtracer.py')
  cpu = cpu[~rem]
  timestamp = cpu['timestamp']
  #name = cpu[['filename', 'function', 'line']].apply(lambda f: sys.intern(f"{f[0]}\x00{f[1]}\x00{f[2]}"), axis=1).rename('name')
  tpe = cpu['what'].apply({'call':'start', 'return':'end', 'ccall':'start', 'cexc':'end', 'cret':'end'}.get).rename('type')
  return pd.concat([cpu.timestamp, cpu.filename, cpu.function, cpu.line, tpe], axis=1)


def from_profiles(cpu_filename, cuda_filename):
  
  #cpu = pd.read_csv(CPU_FILENAME)
  with sqlite3.connect(cpu_filename) as conn:
    #filenames   = pd.read_sql_query("SELECT * FROM FILENAMES", conn, index_col="id")
    #functions   = pd.read_sql_query("SELECT * FROM FUNCTIONS", conn, index_col="id")
    #trace_types = pd.read_sql_query("SELECT * FROM TRACE_TYPES", conn, index_col="id")
    #records     = pd.read_sql_query("SELECT * FROM RECORDS", conn, index_col="id")
    cpu = pd.read_sql_query("SELECT r.id, f.value as filename, fn.value as function, r.line, t.value as what, r.timestamp\
      FROM RECORDS as r INNER JOIN FILENAMES as f ON r.filename = f.id\
      INNER JOIN FUNCTIONS as fn ON r.function = fn.id\
      INNER JOIN TRACE_TYPES as t ON r.what = t.id", conn, index_col="id")
  print("transforming cpu data...")
  cpu = transform_cpu(cpu)
  print("done")
  
  with sqlite3.connect(cuda_filename) as conn:
      cuda_start = pd.read_sql_query("SELECT markers.id as event_id, markers.timestamp, names.value as function FROM CUPTI_ACTIVITY_KIND_MARKER as markers INNER JOIN StringTable as names on markers.name = names._id_ WHERE markers.flags = 2", conn, index_col="event_id")
  
      cuda_end = pd.read_sql_query("SELECT markers.id as event_id, markers.timestamp FROM CUPTI_ACTIVITY_KIND_MARKER as markers  WHERE markers.flags = 4", conn, index_col="event_id").join(cuda_start['function'])
  
  cuda_start['type'] = 'start'
  cuda_end['type'] = 'end'
  cuda_start['filename'] = '<cuda>'
  cuda_end['filename'] = '<cuda>'
  cuda_start['line'] = 0
  cuda_end['line'] = 0
  cuda_start = cuda_start[['timestamp', 'filename', 'function', 'line', 'type']]
  cuda_end = cuda_end[['timestamp', 'filename', 'function', 'line', 'type']]
  cuda = pd.concat([cpu, cuda_start, cuda_end]).sort_values('timestamp')
  cuda['timestamp'] = cuda['timestamp'].diff()
  cuda['timestamp'].iloc[0] = 0
  
  
  root = aggregate(cuda)
  return root

def main(cpu_file=None, cuda_file=None, output=None, json_file=None, show_all=False):
  if (json_file is not None) == (cpu_file is not None and cuda_file is not None):
    print("Usage:\n\tmerge.py --cpu-file cpu.db --cuda-file cuda.prof --output out.json\n\tmerge.py --json-file out.json")
    return

  if json_file is None:
    root = from_profiles(cpu_file, cuda_file)
    if output is not None:
      with open(output, 'w') as f:
        f.write(MyEncoder().encode(root))
  else:
    with open(json_file, 'r') as f:
      j = json.load(f)
      root = from_json(j)
  
  print("framing...")
  root = to_frame(root)
  print("framed")
  
  class DummySession:
    def root_frame(self):
      return root
  
  session = DummySession()
  session.start_time = 0
  session.duration = 0
  session.sample_count = 0
  session.cpu_time = 0
  session.program = "program"
  
  renderer = ConsoleRenderer(unicode=True, color=True, show_all=show_all)
  #renderer.processors.remove(processors.remove_irrelevant_nodes)
  r = renderer.render(session)
  print(r)


if __name__ == '__main__':
  fire.Fire(main)
