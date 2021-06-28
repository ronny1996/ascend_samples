import paddle
import numpy as np

if paddle.fluid.core.is_compiled_with_npu():
  paddle.fluid.framework._set_dygraph_tracer_expected_place(paddle.NPUPlace(0))

def update_params(momentum, linear):
  for i in range(10):
    inp = paddle.full(
        shape=[2, 2], fill_value=i, dtype='float32').astype("float32")
    inp = paddle.to_tensor(inp)
    out = linear(inp)
    loss = paddle.mean(out)
    loss.backward()
    momentum.minimize(loss)
    linear.clear_gradients()

linear_old = paddle.nn.Linear(
    2,
    2,
    weight_attr=paddle.nn.initializer.Constant(value=2.0),
    bias_attr=paddle.nn.initializer.Constant(value=2.0))

momentum_old = paddle.fluid.optimizer.Momentum(
    learning_rate=0.01,
    momentum=0.9,
    parameter_list=linear_old.parameters(),
    regularization=paddle.fluid.regularizer.L2Decay(
        regularization_coeff=0.1))

print(linear_old.weight.numpy())
update_params(momentum_old, linear_old)
print(linear_old.weight.numpy())
