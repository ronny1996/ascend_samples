import paddle
import numpy as np
from paddle import nn

if paddle.fluid.core.is_compiled_with_npu():
  paddle.fluid.framework._set_dygraph_tracer_expected_place(paddle.NPUPlace(0))

"""
[[[0, 1],
  [2, 3]],
 [[4, 5],
  [6, 7]],
 [[8, 9],
  [10, 11]]],

[[[12, 13],
  [14, 15]],
 [[16, 17],
  [18, 19]],
 [[20, 21],
  [22, 23]]]
"""
x_np = np.array([x for x in range(2*3*4)]).reshape([2, 3, 2, 2]).astype(np.float32)
x_var = paddle.to_tensor(x_np, stop_gradient=False)
y_var = paddle.fluid.layers.reduce_mean(input=x_var, dim=[0, 2], keep_dim=True)
print(y_var)
out = y_var.sum()
out.backward()
print(x_var.grad)

# y_np = np.transpose(x_np, axes=[0, 2, 1, 3]).reshape(-1, x_np.shape[1], x_np.shape[3])
# y_np = np.mean(y_np, 0)
# print(y_np)