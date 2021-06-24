import paddle
import numpy as np
from paddle import nn

paddle.fluid.framework._set_dygraph_tracer_expected_place(paddle.NPUPlace(0))

x_np = np.ones([1, 1, 3, 3]).astype(np.float32)
x_var = paddle.to_tensor(x_np)
conv = nn.Conv2D(1, 1, (2, 2), stride=1, padding=2)
y_var = conv(x_var)
y_np = y_var.numpy()
print(y_np)

# print(conv.weight)
# print(conv.bias)

s = y_var.sum()
s.backward()

print(conv.weight.grad)
print(conv.bias.grad)
