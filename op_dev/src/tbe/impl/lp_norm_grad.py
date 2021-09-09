import tbe
from tbe import tvm
from tbe import dsl
from tbe.common.register import register_op_compute
from tbe.common.utils import shape_util, para_check


# def get_lp_norm_output_shape(x_shape, axes=[], keepdims=False):
#     y_shape = []
#     for ax in range(len(x_shape)):
#         if (len(axes) == 0 or ax in axes) and keepdims:
#             y_shape.append(1)
#         else:
#             y_shape.append(x_shape[ax])                
#     return y_shape

@register_op_compute("lp_norm_grad")
def lp_norm_grad_compute(x, y, y_grad, x_grad, p=1, axes=[], epsilon=1e-12, keepdims=False, kernel_name="lp_norm_grad"):
    x_shape = shape_util.shape_to_list(x.shape)
    x_dtype = x.dtype.lower()
    y_shape = shape_util.shape_to_list(y.shape)
    y_dtype = y.dtype.lower()
    dy_shape = shape_util.shape_to_list(y_grad.shape)
    dy_dtype = y_grad.dtype.lower()

    p_val = tvm.const(p, x_dtype)
    epsilon_val = tvm.const(epsilon, x_dtype)

    # y = y.reshape(tuple(y_shape))
    # y_grad = y_grad.reshape(tuple(y_shape))

    # if (porder == 0) {
    #   set_zero(dev_ctx, out_dx, static_cast<T>(0));
    # } else if (porder == INFINITY || porder == -INFINITY) {
    #   dx.device(*place) =
    #       (x.abs() == norm.broadcast(bcast)).template cast<T>() * x.sign() *
    #       norm_dy.broadcast(bcast);
    # } else {
    #   dx.device(*place) =
    #       (x.abs()).pow(porder - 1.0f) /
    #       ((norm.broadcast(bcast)).pow(porder - 1.0f) + x.constant(eps));
    #   dx.device(*place) = dx * norm_dy.broadcast(bcast) * x.sign();
    # }

    # if p == 0:
    #     res = dsl.vcmp(x, tvm.const(0, x_dtype), 'ne', 'bit')
    #     res = dsl.cast_to(res, x_dtype)
    #     res = dsl.reduce_sum(res, axis=1, keepdims=keepdims)

    # elif p == float("inf") or p == float("-inf"):
    #     abs_x = dsl.vabs(x)
    #     bcast_y = dsl.broadcast(y, x_shape)
    #     bcast_dy = dsl.broadcast(y_grad, x_shape)
    #     res = dsl.vcmp(abs_x, bcast_y, 'eq', 'bit')
    #     res = dsl.cast_to(res, x_dtype)
    #     res = dsl.vmul(res, bcast_dy)

    # else:
    #     abs_x = dsl.vabs(x)
    #     bcast_y = dsl.broadcast(y, x_shape)
    #     bcast_dy = dsl.broadcast(y_grad, x_shape)
    #     res = dsl.vmul(bcast_y, bcast_dy)
    res = dsl.vabs(y)

    return res

@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
def lp_norm_grad(x, y, y_grad, x_grad, p=1, axes=[], epsilon=1e-12, keepdims=False, kernel_name="lp_norm_grad"):
    x_shape = x.get("shape")
    y_shape = y.get("shape")
    y_grad_shape = y_grad.get("shape")

    x_dtype = x.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    y_grad_dtype = y_grad.get("dtype").lower()

    data_x = tvm.placeholder(x_shape, dtype=x_dtype, name="data_x")
    data_y = tvm.placeholder(y_shape, dtype=y_dtype, name="data_y")
    data_y_grad = tvm.placeholder(y_grad_shape, dtype=y_grad_dtype, name="data_y_grad")

    res = lp_norm_grad_compute(data_x, data_y, data_y_grad, x_grad, p, axes, epsilon, keepdims, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = dsl.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, data_y_grad, res]}
    dsl.build(schedule, config)
    