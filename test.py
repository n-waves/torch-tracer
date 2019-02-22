import torch
from torch import nn


class MultTest(nn.Module):
  def __init__(self):
    super(MultTest, self).__init__()
    self.L1 = nn.Linear(1000, 1000)
    self.L2 = nn.Linear(1000, 1000)
  def first(self, x):
    for i in range(200):
      x = self.L1(x)
    return x
  def second(self, x):
    for i in range(500):
      x = self.L2(x)
    return x
  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x

with torch.autograd.profiler.emit_nvtx():
  mult = MultTest().cuda()
  for i in range(10):
    x = torch.randn(1000, 1000).cuda()
    y = mult(x)
    s = y.sum()
    s.backward()
