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
# x_np = np.array([x for x in range(2 * 4 * 4 * 2)]).reshape([2, 2, 4, 4]).astype(np.float32)
x_np = np.random.random([2, 3, 5, 5]).astype(np.float32)
x_var = paddle.to_tensor(x_np, stop_gradient=False)
pool = nn.AvgPool2D(kernel_size=(3, 3), stride=1, padding=0, data_format="NCHW")
y_var = pool(x_var)
# print(y_var.numpy())
s = y_var.sum()
s.backward()
print(x_var.grad.numpy().squeeze())
