import paddle
import numpy as np
from paddle import nn

if paddle.fluid.core.is_compiled_with_npu():
  paddle.fluid.framework._set_dygraph_tracer_expected_place(paddle.NPUPlace(0))

"""
[[0, 1, 2, 3],
[4, 5, 6, 7],
[8, 9, 10, 11]],

[[0, 1, 2, 3],
[4, 5, 6, 7],
[8, 9, 10, 11]] + 12
"""
x_np = np.array([x for x in range(2*3*4)]).reshape([2, 3, 4]).astype(np.float32)
x_var = paddle.to_tensor(x_np, stop_gradient=False)
y_var = paddle.fluid.layers.reduce_mean(input=x_var, dim=[1, 2])
print(y_var)
