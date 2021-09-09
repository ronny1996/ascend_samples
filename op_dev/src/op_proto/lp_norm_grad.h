#ifndef GE_OP_LP_NORM_GRAD_H
#define GE_OP_LP_NORM_GRAD_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(LpNormGrad)
    .INPUT(x, TensorType({DT_FLOAT32,DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT32,DT_FLOAT16}))
    .INPUT(y_grad, TensorType({DT_FLOAT32,DT_FLOAT16}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT32,DT_FLOAT16}))
    .ATTR(p, Float, 1.0)
    .ATTR(axes, ListInt, {})
    .ATTR(epsilon, Float, 1e-12)
    .ATTR(keepdims, Bool, false)
    .OP_END_FACTORY_REG(LpNormGrad)
}
#endif //GE_OP_LP_NORM_GRAD_H
