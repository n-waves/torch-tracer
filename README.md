Requirements:

```conda install cython pandas numba && pip install fire pyinstrument && pip install -e torch-tracer/recorder```

Test:

```CUDA_LAUNCH_BLOCKING=1 /usr/local/cuda-9.1/bin/nvprof --profile-from-start off -f -o cuda.prof -- python torch-tracer/torchtracer.py torch-tracer/test.py```

This will create `cpu.db` (hardcoded) and `cuda.prof`.

To see the results:

```python torch-tracer/merge.py --cpu-file cpu.db --cuda-file cuda.prof --output out.json```

```
   ├─ 4.703 backward  torch/tensor.py:74
   │     [10 frames hidden]  torch
   │        4.702 backward  torch/autograd/__init__.py:38
   │        ├─ 2.604 AddmmBackward (addmm:1)  ../<cuda>:0
   │        │  └─ 2.499 mm:0  ../<cuda>:0
   │        ├─ 1.019 AddmmBackward (addmm:0)  ../<cuda>:0
   │        │  └─ 0.978 mm:1  ../<cuda>:0
   │        ├─ 0.532 add:0  ../<cuda>:0
   │        ├─ 0.295 sum:1  ../<cuda>:0
   ├─ 2.639 __call__  torch/nn/modules/module.py:483
   │     [4 frames hidden]  torch
   │        2.639 forward  test.py:18
   │        ├─ 1.891 second  test.py:14
   │        │  └─ 1.862 __call__  torch/nn/modules/module.py:483
   │        │        [14 frames hidden]  torch
   │        │           1.797 linear  torch/nn/functional.py:1336
   │        │           └─ 1.728 addmm:1  ../<cuda>:0
   │        └─ 0.747 first  test.py:10
   │           └─ 0.735 __call__  torch/nn/modules/module.py:483
   │                 [14 frames hidden]  torch
   │                    0.707 linear  torch/nn/functional.py:1336
   │                    └─ 0.678 addmm:0  ../<cuda>:0
```

The aggregating part of `merger.py` can take a lot of time to finish, usually much longer than the original script that was profiled. You can use a c++ implementation of aggregation to get the same results. F.e., on ubuntu 18.04:

```
sudo apt-get install libsqlite3-dev
mkdir torch-tracer/bin
g++ torch-tracer/aggregate.cpp -o torch-tracer/bin/aggregate --std=c++17 -l sqlite3 -O2
```

and then:

```
torch-tracer/bin/aggregate cpu.db cuda.prof out.json
python torch-tracer/merge.py --json-file out.json
```

CUDA operations in forward and backward passes can be matched by the sequence number. F.e., `addmm: 0` in backward pass is a result of `linear` called in `first()`.

Profiler overhead:
 * cuda launch blocking
 * record buffer resizing
 * recording
