import tbe
from tbe import tvm
from tbe import dsl
from tbe.common.register import register_op_compute
from tbe.common.utils import shape_util


@register_op_compute("axpb")
def axpb_compute(x, y, alpha=1, beta=0, kernel_name="axpb"):
    x_shape = shape_util.shape_to_list(x.shape)
    x_dtype = x.dtype.lower()

    alpha_val = tvm.const(alpha, x_dtype)
    beta_val = tvm.const(beta, x_dtype)
    x = dsl.vmuls(x, alpha_val)
    res = dsl.vadds(x, beta_val)
    return res

def axpb(x, y, alpha=1, beta=0, kernel_name="axpb"):
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = axpb_compute(data_x, y, alpha, beta, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = dsl.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    dsl.build(schedule, config)
    