import paddle
import numpy as np

if paddle.fluid.core.is_compiled_with_npu():
  paddle.fluid.framework._set_dygraph_tracer_expected_place(paddle.NPUPlace(0))

x_np = np.array(list(range(0, 3 * 4 * 2 * 2))).reshape([3, 4, 2, 2]).astype(np.float32)
# x_np = np.ones([3, 4, 1, 2]).astype(np.float32)

y_np = np.array([x for x in range(4 * 1 * 2)]).reshape([4, 1, 2]).astype(np.float32)

x_var = paddle.to_tensor(x_np, stop_gradient=False)
y_var = paddle.to_tensor(y_np, stop_gradient=False)

out = paddle.fluid.layers.nn.elementwise_add(x_var, y_var, axis=1)
print(out)
s = out.sum()
s.backward()
print(out.grad)
print(x_var.grad)
print(y_var.grad)
