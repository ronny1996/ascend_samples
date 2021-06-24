import paddle
import numpy as np

paddle.fluid.framework._set_dygraph_tracer_expected_place(paddle.NPUPlace(0))


def _reference_testing(x, scale, offset, mean, var, epsilon, data_format):
    x_shape = x.shape
    if len(x_shape) == 2:
        if data_format == "NCHW":
            x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
        else:
            x = np.reshape(x, (x.shape[0], 1, 1, x.shape[1]))

    if data_format == "NCHW":
        n, c, h, w = x.shape
        mean_tile = np.reshape(mean, (1, c, 1, 1))
        mean_tile = np.tile(mean_tile, (n, 1, h, w))
        var_tile = np.reshape(var, (1, c, 1, 1))
        var_tile = np.tile(var_tile, (n, 1, h, w))
        normalized = (x - mean_tile) / np.sqrt(var_tile + epsilon)
        scale_tile = np.reshape(scale, (1, c, 1, 1))
        scale_tile = np.tile(scale_tile, (n, 1, h, w))
        offset_tile = np.reshape(offset, (1, c, 1, 1))
        offset_tile = np.reshape(offset_tile, (1, c, 1, 1))
        y = normalized * scale_tile + offset_tile
    elif data_format == "NHWC":
        normalized = (x - mean) / np.sqrt(var + epsilon)
        y = normalized * scale + offset
    else:
        raise ValueError("Unknown data order.")

    if len(x_shape) == 2:
        y = np.reshape(y, x_shape)
    return y


def _cal_mean_variance(x, epsilon, data_format):
    assert data_format in ['NCHW', 'NHWC']
    x_square = x * x
    axis = (0, 2, 3) if data_format == 'NCHW' else (0, 1, 2)
    C = x.shape[1] if data_format == 'NCHW' else x.shape[-1]
    x_square_sum = np.sum(x_square, axis)
    x_sum = np.sum(x, axis=axis)
    element_count = np.size(x) / C
    mean = x_sum / element_count
    var = x_square_sum / element_count - mean * mean
    return mean, var


def _reference_training(x, scale, offset, epsilon, data_format):
    x_shape = x.shape
    if data_format == "NCHW":
        n, c, h, w = x.shape
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 2, 3))
        x_sum = np.sum(x, axis=(0, 2, 3))
        element_count = np.size(x) / int(np.shape(x)[1])
        mean = x_sum / element_count
        var = x_square_sum / element_count - mean * mean
        mean_tile = np.reshape(mean, (1, c, 1, 1))
        mean_tile = np.tile(mean_tile, (n, 1, h, w))
        var_tile = np.reshape(var, (1, c, 1, 1))
        var_tile = np.tile(var_tile, (n, 1, h, w))
        normalized = (x - mean_tile) / np.sqrt(var_tile + epsilon)
        scale_tile = np.reshape(scale, (1, c, 1, 1))
        scale_tile = np.tile(scale_tile, (n, 1, h, w))
        offset_tile = np.reshape(offset, (1, c, 1, 1))
        offset_tile = np.reshape(offset_tile, (1, c, 1, 1))
        y = normalized * scale_tile + offset_tile
        return y, mean, var
    elif data_format == "NHWC":
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 1, 2))
        x_sum = np.sum(x, axis=(0, 1, 2))
        element_count = np.size(x) / int(np.shape(x)[-1])
        mean = x_sum / element_count
        var = x_square_sum / element_count - mean * mean
        normalized = (x - mean) / np.sqrt(var + epsilon)
        y = normalized * scale + offset
        return y, mean, var
    else:
        raise ValueError("Unknown data order.")


def _reference_grad(x, y_grad, scale, mean, var, epsilon, data_format):
    if data_format != "NCHW" and data_format != "NHWC":
        raise ValueError("Unknown data order.")

    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))
        y_grad = np.transpose(y_grad, (0, 2, 3, 1))

    x_grad = scale * (y_grad - np.mean(
        y_grad, axis=(0, 1, 2)) - (x - mean) * np.mean(
            y_grad * (x - mean), axis=(0, 1, 2)) /
        (var + epsilon)) / np.sqrt(var + epsilon)
    grad_scale = np.sum(y_grad * (x - mean) / np.sqrt(var + epsilon),
                        axis=(0, 1, 2))
    grad_offset = np.sum(y_grad, axis=(0, 1, 2))

    # transfer back to N, C, H, W
    if data_format == "NCHW":
        x_grad = np.transpose(x_grad, (0, 3, 1, 2))
        x = np.transpose(x, (0, 3, 1, 2))
        y_grad = np.transpose(y_grad, (0, 3, 1, 2))

    return x_grad, grad_scale, grad_offset


x_np = np.random.normal(loc=1, scale=3, size = [2, 3, 2, 2]).astype(np.float32)
x = paddle.to_tensor(x_np, stop_gradient=False)
bn = paddle.nn.BatchNorm(num_channels=3, momentum=0)
scale = bn.weight.numpy()
bias = bn.bias.numpy()
y_ref = _reference_training(x_np, scale, bias, bn._epsilon, "NCHW")

y = bn(x)
y_sum = y.sum()
print(y_ref[0])
print(y.numpy())
print(y_ref[1])
print(bn._mean.numpy())
print(y_ref[2])
print(bn._variance.numpy())


y_sum.backward()
y_grad_ref = _reference_grad(x_np, y.grad.numpy(), scale, bn._mean.numpy(), bn._variance.numpy(), bn._epsilon, "NCHW")
print(x.grad.numpy())
print(y_grad_ref[0])
print(bn.weight.grad.numpy())
print(y_grad_ref[1])
print(bn.bias.grad.numpy())
print(y_grad_ref[2])
np.allclose(y_ref[0], y.numpy())
np.allclose(y_ref[1], bn._mean.numpy())
np.allclose(y_ref[2], bn._variance.numpy())

