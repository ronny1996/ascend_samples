import paddle
import numpy as np
from paddle import nn

if paddle.fluid.core.is_compiled_with_npu():
  paddle.fluid.framework._set_dygraph_tracer_expected_place(paddle.NPUPlace(0))

"""
1 2 3
4 5 6
7 8 9
"""
x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape([1, 1, 3, 3]).astype(np.float32)
x_var = paddle.to_tensor(x_np, stop_gradient=False)
pool = nn.AvgPool2D(kernel_size=(2, 2), stride=2, padding=0)
y_var = pool(x_var)
print(y_var.numpy())
pool = nn.AvgPool2D(kernel_size=(2, 2), stride=1, padding=0)
y_var = pool(x_var)
print(y_var.numpy())
pool = nn.AvgPool2D(kernel_size=(2, 2), stride=2, padding="VALID")
y_var = pool(x_var)
print(y_var.numpy())
# pool = nn.AvgPool2D(kernel_size=(2, 2), stride=2, padding="SAME")
# y_var = pool(x_var)
# print(y_var.numpy())

s = y_var.sum()
s.backward()
print(x_var.grad)