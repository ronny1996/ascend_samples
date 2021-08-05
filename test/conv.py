import paddle
import numpy as np
from paddle import nn

def func(p):
  paddle.fluid.framework._set_dygraph_tracer_expected_place(p)
  np.random.seed(123)
  x_np = np.random.random([32, 512, 7, 7]).astype(np.float32)
  x_var = paddle.to_tensor(x_np)
  conv = nn.Conv2D(512, 2048, (1, 1), stride=1, padding=0)
  y_var = conv(x_var)
  # print(conv.weight)
  # y_np = y_var.numpy()
  # print(conv.weight)
  # print(conv.bias)
  s = y_var.sum()
  s.backward()
  y_var.backward()
  # y_var.backward()
  return conv.weight.grad.numpy().flatten()
# print(conv.bias.grad)
npu = func(paddle.NPUPlace(0))
cpu = func(paddle.CPUPlace())

abs_diff = abs(npu - cpu)
abs_npu = abs(npu)
index = np.argmax(abs_diff / abs_npu)
print(npu[index], cpu[index])
