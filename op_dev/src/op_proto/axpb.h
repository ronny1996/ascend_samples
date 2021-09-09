#ifndef GE_OP_AXPB_H
#define GE_OP_AXPB_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(Axpb)
    .INPUT(x, TensorType({DT_FLOAT32,DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT32,DT_FLOAT16}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .OP_END_FACTORY_REG(Axpb)
}
#endif //GE_OP_AXPB_H
